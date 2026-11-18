"""Native vLLM backend for Qwen3-ASR: vllm.LLM.chat(), no qwen_asr package.

PARTIAL is re-inferred from the audio accumulated for the utterance; this is
*windowed* streaming, NOT token-level streaming. Keep utterances bounded.
"""
import base64
import io
import wave
from dataclasses import dataclass, field

import numpy as np

from .text_clean import split_asr_output


@dataclass
class _VLLMUtteranceState:
    """One RPC utterance; numpy audio lives on CPU, vLLM weights live on GPU."""
    language: str
    chunk_size_samples: int
    chunk_id: int = 0
    text: str = ''          # transcript with markers removed
    raw_output: str = ''    # last raw model answer, markers included
    buffer: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    chunks: list = field(default_factory=list)
    total_samples: int = 0
    last_inferred_samples: int = 0


class VLLMASRBackend:
    """Adapter for Qwen3-ASR using only vllm.LLM.chat().

    The spoken language is auto-detected by the model. `state.language` starts as
    the language the client configured and is replaced by the detected one after
    the first inference that names it.

    The native vLLM chat API transcribes *complete* audio. To provide gRPC
    PARTIAL events, infer again on the utterance accumulated at every chunk.
    No qwen_asr streaming state/KV cache is assumed. Keep utterances bounded.
    """

    def __init__(self, model_path=None, *, gpu_memory_utilization=0.75,
                 max_new_tokens=256, max_model_len=8192, max_num_seqs=1,
                 dtype='float16', enforce_eager=False,
                 llm=None, sampling_params=None):
        if llm is None:
            # Lazy import: unit tests and CPU VAD do not require vLLM.
            from vllm import LLM, SamplingParams
            if model_path is None:
                raise ValueError('model_path must be provided to load vLLM')
            llm = LLM(
                model=str(model_path),
                tokenizer=str(model_path),
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                limit_mm_per_prompt={'audio': 1},
                enforce_eager=enforce_eager,
            )
            sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        elif sampling_params is None:
            # For tests using an injected fake LLM, no vLLM import required.
            sampling_params = object()
        self.llm = llm
        self.sampling_params = sampling_params

    def init_streaming_state(self, *, language='Vietnamese',
                             unfixed_chunk_num=2, unfixed_token_num=5,
                             chunk_size_sec=1.0):
        # The last two args are accepted for compatibility with the gRPC
        # service; they belong to qwen_asr and are unused by vLLM.chat().
        del unfixed_chunk_num, unfixed_token_num
        if chunk_size_sec <= 0:
            raise ValueError('chunk_size_sec must be positive')
        return _VLLMUtteranceState(
            language=language,
            chunk_size_samples=max(1, round(chunk_size_sec * 16000)),
        )

    @staticmethod
    def _wav_data_uri(samples):
        """16-kHz mono PCM16 WAV in memory: no network and no temp files."""
        samples = np.asarray(samples, dtype=np.float32)
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype('<i2', copy=False)
        target = io.BytesIO()
        with wave.open(target, 'wb') as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(pcm.tobytes())
        return 'data:audio/wav;base64,' + base64.b64encode(target.getvalue()).decode('ascii')

    def _infer(self, state):
        if not state.total_samples:
            return
        audio = np.concatenate(state.chunks)
        conversation = [{'role': 'user', 'content': [
            {'type': 'audio_url', 'audio_url': {'url': self._wav_data_uri(audio)}}
        ]}]
        outputs = self.llm.chat(conversation, sampling_params=self.sampling_params, use_tqdm=False)
        # Raw answer looks like `language Vietnamese<asr_text>xin chao`: keep the text,
        # and take the language the model detected from the marker.
        state.raw_output = outputs[0].outputs[0].text.strip()
        language, state.text = split_asr_output(state.raw_output)
        if language:
            state.language = language
        state.last_inferred_samples = state.total_samples
        state.chunk_id += 1

    def streaming_transcribe(self, samples, state):
        samples = np.asarray(samples, dtype=np.float32)
        if not samples.size:
            return state
        state.chunks.append(samples.copy())
        state.total_samples += samples.size
        state.buffer = np.concatenate((state.buffer, samples))
        if state.buffer.size >= state.chunk_size_samples:
            # Consumed N complete intervals; leftover is the next interval.
            consumed = (state.buffer.size // state.chunk_size_samples) * state.chunk_size_samples
            state.buffer = state.buffer[consumed:]
            self._infer(state)
        return state

    def finish_streaming_transcribe(self, state):
        # Ensure short utterances (< chunk_sec) and remaining tail are
        # recognized, but do not redundantly re-infer unchanged complete audio.
        if state.total_samples > state.last_inferred_samples:
            self._infer(state)
        return state
