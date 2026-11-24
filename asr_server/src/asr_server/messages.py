"""Protocol-independent requests and results between the gRPC layer and the core.

The core (server.ASRService) knows nothing about speech.proto: the gRPC layer
(speech_service.SpeechStreamService) turns AudioChunk messages into these
requests and turns Transcript results into RecognitionResult messages.
"""
from dataclasses import dataclass

# audio encodings accepted by the core
PCMA = 1          # G.711 A-law, 8 kHz
PCMU = 2          # G.711 mu-law, 8 kHz
PCM_S16LE = 3     # little-endian mono signed PCM, 8 or 16 kHz

# transcript events
PARTIAL = 1       # replace the current uncommitted utterance text
COMMIT = 2        # immutable final utterance text

AUTO_LANGUAGE = 'auto'    # StreamConfig.language: the model detects the language


@dataclass
class StreamConfig:
    call_id: str = ''
    leg_id: str = ''
    encoding: int = 0
    sample_rate: int = 0
    language: str = ''        # model language name ('Vietnamese'), AUTO_LANGUAGE; '' = server default


@dataclass
class AudioChunk:
    data: bytes = b''
    sequence: int = 0         # 1-based, +1 per chunk


class EndUtterance:
    """Force a commit of the open utterance; the stream goes on."""


class EndStream:
    """Flush the open utterance, then close."""


class StreamRequest:
    """Exactly one of config / audio / end_utterance / end_stream."""

    __slots__ = ('config', 'audio', 'end_utterance', 'end_stream')

    def __init__(self, config=None, audio=None, end_utterance=None, end_stream=None):
        self.config = config
        self.audio = audio
        self.end_utterance = end_utterance
        self.end_stream = end_stream

    def WhichOneof(self, _name='payload'):
        for name in self.__slots__:
            if getattr(self, name) is not None:
                return name
        return None


@dataclass
class Transcript:
    call_id: str = ''
    leg_id: str = ''
    utterance_index: int = 0
    text: str = ''            # formatted text for THIS utterance
    is_final: bool = False    # true only when event == COMMIT
    revision: int = 0         # monotonic within utterance_index
    language: str = ''        # model language name
    reason: str = ''          # partial / vad_endpoint / semantic_pause / max_duration / end_stream / ...
    event: int = 0
    raw_text: str = ''        # ASR text, markers removed, before punctuation
    commit_id: int = 0        # monotonic per stream, 0 for PARTIAL
    start_ms: int = 0         # media time from the start of the stream
    end_ms: int = 0
