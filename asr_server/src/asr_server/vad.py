"""Server-side VAD endpointers with pre-roll, hangover and optional short-pause event.

Two interchangeable classes with the same events and attributes:

- SileroEndpointer: Silero TorchScript VAD on CPU, 32 ms frames (256 samples at
  8 kHz, 512 at 16 kHz), speech probability with onset/exit hysteresis. Loads the
  local silero_vad.jit with torch.jit.load; no webrtcvad, torch.hub or network.
  IMPORTANT: the Silero model is stateful, so every gRPC stream MUST have its own
  instance. Sharing one mixes recurrent state between streams.
- WebRTCEndpointer: WebRTC VAD, 20 ms frames, binary decision.

No audio is dropped before an utterance is confirmed: pre-roll covers the onset decision.
"""
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class AudioEvent:
    kind: str                # audio, pause_candidate, endpoint, discard
    samples: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    start_ms: int = 0
    end_ms: int = 0


class WebRTCEndpointer:
    def __init__(self, sample_rate, *, frame_ms=20, mode=2,
                 preroll_ms=200, onset_window_ms=100, onset_voiced_ms=60,
                 endpoint_ms=600, pause_ms=300, min_voiced_ms=160,
                 classifier=None):
        if sample_rate not in (8000, 16000):
            raise ValueError('VAD supports PCM 8k or 16k')
        if frame_ms not in (10, 20, 30):
            raise ValueError('WebRTC VAD requires 10/20/30 ms frames')
        if not (0 < pause_ms < endpoint_ms):
            raise ValueError('must have 0 < pause_ms < endpoint_ms')
        self.sample_rate = int(sample_rate)
        self.frame_ms = int(frame_ms)
        self.frame_samples = self.sample_rate * self.frame_ms // 1000
        self.endpoint_frames = max(1, round(endpoint_ms / frame_ms))
        self.pause_frames = max(1, round(pause_ms / frame_ms))
        self.min_voiced_frames = max(1, round(min_voiced_ms / frame_ms))
        self.preroll_frames = max(1, round(preroll_ms / frame_ms))
        self.onset_window = max(1, round(onset_window_ms / frame_ms))
        self.onset_votes = max(1, round(onset_voiced_ms / frame_ms))
        if self.onset_votes > self.onset_window:
            raise ValueError('onset_voiced_ms exceeds onset_window_ms')
        self._classifier = classifier
        self._vad = None
        self.offload = False                  # microseconds per frame: runs inline
        if classifier is None:
            import webrtcvad                  # only needed for this backend
            self._vad = webrtcvad.Vad(mode)
        self.pending = np.empty(0, np.float32)
        self.processed_frames = 0
        self.active = False
        self._reset_activity()

    def _reset_activity(self):
        self.pre_roll = deque(maxlen=self.preroll_frames)
        self.votes = deque(maxlen=self.onset_window)
        self.silence_frames = 0
        self.voiced_frames = 0
        self.pause_sent = False
        self.start_ms = 0

    def _voice(self, samples):
        if self._classifier is not None:
            return bool(self._classifier(samples, self.sample_rate))
        data = (np.clip(samples, -1.0, 1.0) * 32767.0).astype('<i2').tobytes()
        return bool(self._vad.is_speech(data, self.sample_rate))

    def _process_frame(self, frame):
        voice = self._voice(frame)
        self.processed_frames += 1
        media_ms = self.processed_frames * self.frame_ms
        if not self.active:
            self.pre_roll.append((frame.copy(), voice))
            self.votes.append(voice)
            if voice and sum(self.votes) >= self.onset_votes:
                self.active = True
                self.silence_frames = 0
                self.voiced_frames = sum(v for _, v in self.pre_roll)
                self.start_ms = max(0, media_ms - len(self.pre_roll) * self.frame_ms)
                buffered = np.concatenate([f for f, _ in self.pre_roll])
                self.pre_roll.clear()
                self.votes.clear()
                return [AudioEvent('audio', buffered, self.start_ms, media_ms)]
            return []

        if voice:
            self.silence_frames = 0
            self.pause_sent = False
            self.voiced_frames += 1
        else:
            self.silence_frames += 1
        events = [AudioEvent('audio', frame, self.start_ms, media_ms)]
        if not voice and self.silence_frames == self.pause_frames and not self.pause_sent:
            self.pause_sent = True
            events.append(AudioEvent('pause_candidate', start_ms=self.start_ms, end_ms=media_ms))
        if self.silence_frames >= self.endpoint_frames:
            kind = 'endpoint' if self.voiced_frames >= self.min_voiced_frames else 'discard'
            events.append(AudioEvent(kind, start_ms=self.start_ms, end_ms=media_ms))
            self.active = False
            self._reset_activity()
        return events

    def push(self, samples):
        """Return ordered events. Use short push increments for early commit reset."""
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError('expected mono PCM float32')
        if len(samples) == 0:
            return []
        self.pending = np.concatenate([self.pending, samples])
        events = []
        while len(self.pending) >= self.frame_samples:
            frame = self.pending[:self.frame_samples]
            self.pending = self.pending[self.frame_samples:]
            events.extend(self._process_frame(frame))
        return events

    def finish(self):
        """Flush active audio on EOS. Silence with no active utterance is ignored."""
        events = []
        if self.active:
            if self.pending.size:
                events.append(AudioEvent('audio', self.pending.copy(), self.start_ms,
                                         round(1000 * (self.processed_frames * self.frame_samples
                                                       + self.pending.size) / self.sample_rate)))
            elapsed = round(1000 * (self.processed_frames * self.frame_samples
                                    + self.pending.size) / self.sample_rate)
            kind = 'endpoint' if self.voiced_frames >= self.min_voiced_frames else 'discard'
            events.append(AudioEvent(kind, start_ms=self.start_ms, end_ms=elapsed))
        self.pending = np.empty(0, np.float32)
        self.active = False
        self._reset_activity()
        return events

    def rebase_start(self, start_ms):
        """Forced split: the audio that continues belongs to a new utterance starting here."""
        self.start_ms = int(start_ms)

    def reset_activity(self):
        """Call after semantic early commit to start another VAD utterance."""
        self.active = False
        self._reset_activity()


class SileroEndpointer:
    def __init__(self, sample_rate, *, model_path=None, threshold=0.5,
                 neg_threshold=None, preroll_ms=200, onset_window_ms=100,
                 onset_voiced_ms=60, endpoint_ms=600, pause_ms=300,
                 min_voiced_ms=160, classifier=None, model=None):
        if sample_rate not in (8000, 16000):
            raise ValueError('Silero VAD supports only 8k or 16k PCM')
        if not (0 < pause_ms < endpoint_ms):
            raise ValueError('must have 0 < pause_ms < endpoint_ms')
        if not 0 < threshold < 1:
            raise ValueError('threshold must be between 0 and 1')
        if neg_threshold is None:
            neg_threshold = max(threshold - 0.15, 0.01)
        if not 0 <= neg_threshold <= threshold:
            raise ValueError('neg_threshold must be between 0 and threshold')

        self.sample_rate = int(sample_rate)
        self.frame_ms = 32                    # Silero supports only 32 ms frames.
        self.frame_samples = 512 if sample_rate == 16000 else 256
        self.threshold = float(threshold)
        self.neg_threshold = float(neg_threshold)
        self.endpoint_frames = max(1, round(endpoint_ms / self.frame_ms))
        self.pause_frames = max(1, round(pause_ms / self.frame_ms))
        self.min_voiced_frames = max(1, round(min_voiced_ms / self.frame_ms))
        self.preroll_frames = max(1, round(preroll_ms / self.frame_ms))
        self.onset_window = max(1, round(onset_window_ms / self.frame_ms))
        self.onset_votes = max(1, round(onset_voiced_ms / self.frame_ms))
        if self.onset_votes > self.onset_window:
            raise ValueError('onset_voiced_ms exceeds onset_window_ms')

        self._classifier = classifier         # Deterministic test override.
        # The torch forward costs CPU time: the server runs push() on a VAD thread,
        # not on the event loop. Not needed with the test classifier.
        self.offload = classifier is None
        self._model = model
        self._torch = None
        if classifier is None:
            import torch
            self._torch = torch
            if self._model is None:
                if model_path is None or not Path(model_path).is_file():
                    raise FileNotFoundError(
                        f'Silero TorchScript checkpoint not found: {model_path}; '
                        'provide --silero-model /path/to/silero_vad.jit')
                # The model includes recurrent VAD state; NEVER share it across
                # gRPC calls. The full checkpoint is loaded once per stream.
                self._model = torch.jit.load(str(model_path), map_location='cpu')
            self._model.eval()
            self._model.reset_states()

        self.pending = np.empty(0, np.float32)
        self.processed_frames = 0
        self.processed_samples = 0   # Includes EOS partial frames for accurate timestamps.
        self.active = False
        self._reset_activity()

    def _reset_activity(self):
        self.pre_roll = deque(maxlen=self.preroll_frames)
        self.votes = deque(maxlen=self.onset_window)
        self.silence_frames = 0
        self.voiced_frames = 0
        self.pause_sent = False
        self.start_ms = 0

    def _voice(self, frame):
        """Use Silero speech probability with onset/exit hysteresis."""
        if self._classifier is not None:
            probability = float(self._classifier(frame, self.sample_rate))
        else:
            with self._torch.inference_mode():
                tensor = self._torch.from_numpy(np.ascontiguousarray(frame))
                probability = float(self._model(tensor, self.sample_rate).item())
        return probability >= (self.neg_threshold if self.active else self.threshold)

    def _process_frame(self, frame):
        voice = self._voice(frame)
        self.processed_frames += 1
        self.processed_samples += self.frame_samples
        media_ms = round(1000 * self.processed_samples / self.sample_rate)
        if not self.active:
            self.pre_roll.append((frame.copy(), voice))
            self.votes.append(voice)
            if voice and sum(self.votes) >= self.onset_votes:
                self.active = True
                self.silence_frames = 0
                self.voiced_frames = sum(v for _, v in self.pre_roll)
                self.start_ms = max(0, media_ms - len(self.pre_roll) * self.frame_ms)
                buffered = np.concatenate([f for f, _ in self.pre_roll])
                self.pre_roll.clear()
                self.votes.clear()
                return [AudioEvent('audio', buffered, self.start_ms, media_ms)]
            return []

        if voice:
            self.silence_frames = 0
            self.pause_sent = False
            self.voiced_frames += 1
        else:
            self.silence_frames += 1
        events = [AudioEvent('audio', frame, self.start_ms, media_ms)]
        if not voice and self.silence_frames == self.pause_frames and not self.pause_sent:
            self.pause_sent = True
            events.append(AudioEvent('pause_candidate', start_ms=self.start_ms, end_ms=media_ms))
        if self.silence_frames >= self.endpoint_frames:
            kind = 'endpoint' if self.voiced_frames >= self.min_voiced_frames else 'discard'
            events.append(AudioEvent(kind, start_ms=self.start_ms, end_ms=media_ms))
            self.active = False
            self._reset_activity()
        return events

    def push(self, samples):
        """Consume arbitrary PCM chunks and return time-ordered AudioEvents."""
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError('expected mono PCM float32')
        if not samples.size:
            return []
        self.pending = np.concatenate([self.pending, samples])
        events = []
        while self.pending.size >= self.frame_samples:
            frame = self.pending[:self.frame_samples]
            self.pending = self.pending[self.frame_samples:]
            events.extend(self._process_frame(frame))
        return events

    def finish(self):
        """Flush active utterance at EOS without fabricating a 32 ms frame."""
        events = []
        elapsed_ms = round(1000 * (self.processed_samples + self.pending.size)
                           / self.sample_rate)
        if self.active:
            if self.pending.size:
                events.append(AudioEvent('audio', self.pending.copy(), self.start_ms,
                                         elapsed_ms))
            kind = 'endpoint' if self.voiced_frames >= self.min_voiced_frames else 'discard'
            events.append(AudioEvent(kind, start_ms=self.start_ms, end_ms=elapsed_ms))
        self.processed_samples += self.pending.size
        self.pending = np.empty(0, np.float32)
        self.active = False
        self._reset_activity()
        # The caller may send EndUtterance and continue the same stream.
        # Keep the Silero recurrent state and the absolute audio clock.
        return events

    def rebase_start(self, start_ms):
        """Forced split: the audio that continues belongs to a new utterance starting here."""
        self.start_ms = int(start_ms)

    def reset_activity(self):
        """Reset endpoint activity after semantic commit, not Silero history/time."""
        self.active = False
        self._reset_activity()
