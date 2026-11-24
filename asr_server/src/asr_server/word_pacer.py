"""Word-by-word reveal of PARTIAL results (word_reveal.* in config/server.yaml).

The model answers once per asr.chunk_sec, with several new words at a time. WordPacer turns
one such PARTIAL into several smaller ones so the text on screen grows one word at a time.

- The first waiting word goes out at once, the others are spread over window_ms. Two sends
  are at least min_interval_ms apart; when too many words wait, they go out in groups.
- A word the model corrected is replaced on the next send; the text never shrinks unless the
  model's own text became shorter.
- A FINAL or an empty PARTIAL goes out at once and cancels the words still waiting.
- `revision` is renumbered, so it still grows by 1 per result sent for an utterance.

WordPacer holds the rules and takes the time as an argument; paced_transcripts runs it.
"""
import asyncio
import math
import time
from dataclasses import replace

# Results of the core waiting to be paced. When the client is slow the core waits too.
QUEUE_SIZE = 8
_END = object()


class WordPacer:
    def __init__(self, window_ms, min_interval_ms):
        if not 0 < min_interval_ms <= window_ms:
            raise ValueError('require 0 < min_interval_ms <= window_ms')
        self.window_ms = window_ms
        self.min_interval_ms = min_interval_ms
        self._utterance = None
        self._revision = 0
        self._last_send_ms = None
        self._forget()

    def _forget(self):
        """Nothing of the current utterance is on screen and nothing is waiting."""
        self._source = None
        self._words = []
        self._raw_words = []
        self._shown = 0
        self._visible = ('', '')                    # (text, raw_text) last sent
        self._ticks_left = 0
        self._interval_ms = self.min_interval_ms
        self._due_ms = None

    def due_ms(self):
        """Time of the next tick(), None when no word is waiting."""
        return self._due_ms

    def _send(self, transcript, now_ms, **changes):
        self._revision += 1
        self._visible = (changes.get('text', transcript.text),
                         changes.get('raw_text', transcript.raw_text))
        self._last_send_ms = now_ms
        return replace(transcript, revision=self._revision, **changes)

    def _prefix(self):
        """The current PARTIAL cut to the words on screen."""
        if self._shown >= len(self._words):
            return self._source.text, self._source.raw_text
        return ' '.join(self._words[:self._shown]), ' '.join(self._raw_words[:self._shown])

    def _send_prefix(self, now_ms):
        text, raw_text = self._prefix()
        return self._send(self._source, now_ms, text=text, raw_text=raw_text)

    def _can_send(self, now_ms):
        return self._last_send_ms is None or now_ms - self._last_send_ms >= self.min_interval_ms

    def push(self, transcript, now_ms):
        """A result of the core arrived; returns the results to send now."""
        if transcript.utterance_index != self._utterance:
            self._utterance = transcript.utterance_index
            self._revision = 0
            self._forget()
        if transcript.is_final or not transcript.text.strip():
            self._forget()
            return [self._send(transcript, now_ms)]
        words, raw_words = transcript.text.split(), transcript.raw_text.split()
        self._source, self._words, self._raw_words = transcript, words, raw_words
        if len(words) != len(raw_words):            # cannot be cut word by word: send it whole
            self._shown = len(words)
        self._shown = max(1, min(self._shown, len(words)))
        changed = self._prefix() != self._visible
        can_send = self._can_send(now_ms)
        if can_send and not changed and self._shown < len(words):
            self._shown += 1                        # the next word needs no waiting
            changed = True

        waiting = len(words) - self._shown
        delay_ms = 0 if can_send else self._last_send_ms + self.min_interval_ms - now_ms
        room_ms = self.window_ms - delay_ms         # what is left of the window after the wait
        self._ticks_left = min(waiting, max(1, int(room_ms // self.min_interval_ms)))
        self._interval_ms = max(self.min_interval_ms, room_ms / (self._ticks_left + 1))
        if changed and can_send:
            self._due_ms = now_ms + self._interval_ms if waiting else None
            return [self._send_prefix(now_ms)]
        self._due_ms = now_ms + delay_ms if changed or waiting else None
        return []

    def tick(self, now_ms):
        """The time of due_ms() has come; returns the results to send now."""
        if self._due_ms is None or now_ms < self._due_ms:
            return []
        # A change that had to wait for the minimum gap goes out first, without a new word.
        if self._ticks_left and self._prefix() == self._visible:
            waiting = len(self._words) - self._shown
            self._shown += math.ceil(waiting / self._ticks_left)
            self._ticks_left -= 1
        self._due_ms = now_ms + self._interval_ms if self._ticks_left else None
        if self._prefix() == self._visible:
            return []
        return [self._send_prefix(now_ms)]

    def flush(self):
        """The stream ended; returns the text still waiting, if any."""
        self._due_ms = None
        self._ticks_left = 0
        if self._source is None:
            return []
        self._shown = len(self._words)
        if self._prefix() == self._visible:
            return []
        return [self._send_prefix(self._last_send_ms)]


async def paced_transcripts(source, pacer):
    """Yield the results of `source` (the core) through `pacer`.

    The core runs in its own task: it waits for the model for hundreds of milliseconds at a
    time, and the words already known must keep going out meanwhile.
    """
    queue = asyncio.Queue(maxsize=QUEUE_SIZE)

    async def pump():
        try:
            async for transcript in source:
                await queue.put(transcript)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await queue.put(exc)
        else:
            await queue.put(_END)
        finally:
            await source.aclose()

    def now_ms():
        return time.monotonic() * 1000.0

    pump_task = asyncio.ensure_future(pump())
    get_task = None
    try:
        while True:
            if pump_task.cancelled() and queue.empty():
                raise asyncio.CancelledError()      # the core was cancelled: nothing more will come
            if get_task is None:
                get_task = asyncio.ensure_future(queue.get())
            due = pacer.due_ms()
            timeout = None if due is None else max(0.0, (due - now_ms()) / 1000.0)
            waiting_for = {get_task} if pump_task.done() else {get_task, pump_task}
            done, _ = await asyncio.wait(waiting_for, timeout=timeout,
                                         return_when=asyncio.FIRST_COMPLETED)
            if not done:
                for transcript in pacer.tick(now_ms()):
                    yield transcript
                continue
            if get_task not in done:
                continue
            item = get_task.result()
            get_task = None
            if item is _END:
                for transcript in pacer.flush():
                    yield transcript
                return
            if isinstance(item, Exception):
                raise item
            for transcript in pacer.push(item, now_ms()):
                yield transcript
    finally:
        pump_task.cancel()
        if get_task is not None:
            get_task.cancel()
        await asyncio.gather(pump_task, return_exceptions=True)
