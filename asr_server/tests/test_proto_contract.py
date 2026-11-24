"""speech.proto contract with the gateway (media_ai/internal/ai/proto/speech.proto).

The gateway encodes/decodes by tag with hand-written protowire code, so tag numbers and wire
bytes are checked here, not only the generated classes.
"""
from pathlib import Path

from asr_server.proto import speech_pb2 as pb

ROOT = Path(__file__).resolve().parents[1]
SERVER_PROTO = ROOT / 'src' / 'asr_server' / 'proto'
EVAL_DIR = ROOT / 'tools' / 'eval'


def _numbers(message):
    return {f.name: f.number for f in message.DESCRIPTOR.fields}


def test_audio_format_enum_matches_the_gateway():
    assert dict(pb.AudioFormat.items()) == {'LEGACY_ENCODED': 0, 'PCM_S16LE': 1}


def test_audio_chunk_tags_match_the_gateway():
    assert _numbers(pb.AudioChunk) == {
        'session_id': 1, 'stream_id': 2, 'pcm': 3, 'sample_rate': 4, 'channels': 5, 'rtp_timestamp': 6,
        'duration_ms': 7, 'end_of_stream': 8, 'language': 9, 'task': 10, 'rtp_clock_rate': 11,
        'chunk_seq': 12, 'language_source': 13, 'output_audio_format': 14,
    }
    assert pb.AudioChunk.DESCRIPTOR.fields_by_name['output_audio_format'].enum_type.name == 'AudioFormat'


def test_result_audio_format_tags_15_16():
    fields = pb.RecognitionResult.DESCRIPTOR.fields_by_name
    assert (fields['audio_format'].number, fields['audio_sample_rate'].number) == (15, 16)
    assert fields['audio_format'].enum_type.name == 'AudioFormat'
    assert fields['audio_sample_rate'].type == fields['audio_sample_rate'].TYPE_INT32


def test_output_audio_format_bytes_from_the_gateway():
    """rtpgw writes field 14 as varint 1 for G.722/L16 sessions and omits it for G.711."""
    base = b'\x0a\x04sess'                                   # session_id = "sess"
    assert pb.AudioChunk.FromString(base + b'\x70\x01').output_audio_format == pb.PCM_S16LE
    assert pb.AudioChunk.FromString(base).output_audio_format == pb.LEGACY_ENCODED


def test_result_audio_format_wire_bytes_are_what_the_gateway_reads():
    """Tag 15 varint 1, tag 16 varint 16000: 78 01 | 80 01 80 7d."""
    result = pb.RecognitionResult(audio_format=pb.PCM_S16LE, audio_sample_rate=16000)
    assert result.SerializeToString() == bytes.fromhex('7801' '8001807d')
    assert pb.RecognitionResult().SerializeToString() == b''   # LEGACY_ENCODED / 0 stay off the wire


def test_eval_client_copies_are_in_sync():
    """tools/eval ships its own copy of the proto and the generated module (scripts/build_proto.py)."""
    assert (EVAL_DIR / 'speech.proto').read_bytes() == (SERVER_PROTO / 'speech.proto').read_bytes()
    assert (EVAL_DIR / 'speech_pb2.py').read_bytes() == (SERVER_PROTO / 'speech_pb2.py').read_bytes()
