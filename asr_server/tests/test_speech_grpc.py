"""speech.proto layer (SpeechStream.Recognize): field mapping and contract checks.

The core is the real ASRService with fake ASR / punctuation models; see
test_regression.py for the helpers.
"""
import asyncio
import logging

import grpc
import numpy as np
import pytest

from asr_server.proto import speech_pb2 as pb
from asr_server.speech_service import SpeechStreamService

from test_regression import Aborted, FakeASR, FakeContext, make_service, pcm, with_server

CHUNK = 2560            # 80 ms of PCM16 mono at 16 kHz
ONE = [pcm(1.2, True), pcm(0.8, False)]
TWO = ONE + ONE


def chunks(frames, *, session='call-1', stream='call-1-a2b', rtp_start=0, clock=8000, seq_start=0,
           language='vi', task='speech_to_text', end=True, chunk_bytes=CHUNK, sample_rate=16000,
           channels=1, language_source='', mutate=None):
    """List of AudioChunk messages for `frames` (PCM16 bytes)."""
    data = b''.join(frames)
    ticks = (chunk_bytes // 2) * (clock or sample_rate) // sample_rate
    pieces = [data[p:p + chunk_bytes] for p in range(0, len(data), chunk_bytes)]
    out = []
    for n, piece in enumerate(pieces):
        chunk = pb.AudioChunk(
            session_id=session, stream_id=stream, pcm=piece, sample_rate=sample_rate,
            channels=channels, rtp_timestamp=(rtp_start + n * ticks) % (1 << 32),
            duration_ms=len(piece) // 2 * 1000 // sample_rate, rtp_clock_rate=clock,
            chunk_seq=(-1 if seq_start is None else seq_start + n), language=language, task=task,
            language_source=language_source, end_of_stream=(end and n == len(pieces) - 1))
        out.append(chunk)
    if mutate:
        mutate(out)
    return out


async def as_stream(messages):
    for message in messages:
        yield message


def recognise(messages, core=None, **service_options):
    """Run one Recognize call directly (no network); returns the results."""
    service = SpeechStreamService(core or make_service(), **service_options)

    async def run():
        return [r async for r in service.Recognize(as_stream(messages), FakeContext())]
    return asyncio.run(run())


def rejected(messages, **service_options):
    try:
        recognise(messages, **service_options)
    except Aborted as err:
        return err.code
    raise AssertionError('expected the RPC to be aborted')


# ----------------------------------------------------------------- field mapping
def test_partial_then_final_fields():
    out = recognise(chunks(ONE))
    partials = [r for r in out if not r.is_final]
    finals = [r for r in out if r.is_final]
    assert partials and len(finals) == 1 and out[-1].is_final
    final = finals[0]
    assert (final.session_id, final.stream_id) == ('call-1', 'call-1-a2b')
    assert final.text == 'xin chao anh?' and final.raw_text == 'xin chao anh'
    assert final.reason == pb.COMMIT_REASON_SILENCE
    assert final.language == 'vi'
    assert [r.seq for r in out] == list(range(1, len(out) + 1))            # 1-based, no holes
    assert {r.utterance_id for r in out} == {1}
    assert [r.revision for r in out] == list(range(1, len(out) + 1))
    for r in out:
        assert r.confidence == 0.0 and r.audio_payload == b'' and r.audio_pt == 0
        assert r.status == pb.RESULT_STATUS_OK
    assert all(r.reason == pb.COMMIT_REASON_UNSPECIFIED for r in partials)
    assert all(r.raw_text == 'xin chao anh' and r.text == 'xin chao anh?' for r in partials)


def test_two_utterances_ids_and_order():
    out = recognise(chunks(TWO))
    finals = [r for r in out if r.is_final]
    assert [r.utterance_id for r in finals] == [1, 2]
    assert finals[0].ts_end < finals[1].ts_start
    # every PARTIAL belongs to the next FINAL
    ids = [r.utterance_id for r in out]
    assert ids == sorted(ids)
    assert [r.seq for r in out] == list(range(1, len(out) + 1))
    first_of_second = next(r for r in out if r.utterance_id == 2)
    assert first_of_second.revision == 1


def test_silence_gives_no_result_and_no_model_call():
    model = FakeASR(text='xin chao anh')
    assert recognise(chunks([pcm(3.0, False)]), make_service(model)) == []
    assert model.infer_calls == 0 and model.states == 0


def test_semantic_early_commit_reason():
    core = make_service(chunk_sec=0.2, semantic_early_commit=True, semantic_confidence=0.9,
                        stable_updates=2)
    out = recognise(chunks([pcm(1.0, True), pcm(0.35, False)]), core)
    finals = [r for r in out if r.is_final]
    assert len(finals) == 1 and finals[0].reason == pb.COMMIT_REASON_SEMANTIC_PAUSE


def test_forced_split_reason_and_contiguous_ticks():
    out = recognise(chunks([pcm(3.0, True), pcm(0.8, False)]), make_service(max_utterance_sec=1.0))
    finals = [r for r in out if r.is_final]
    assert [r.reason for r in finals[:3]] == [pb.COMMIT_REASON_MAX_DURATION] * 3
    assert (finals[0].ts_start, finals[0].ts_end) == (0, 8000)             # 1000 ms at 8 kHz
    for prev, cur in zip(finals, finals[1:]):
        assert cur.ts_start == prev.ts_end


def test_empty_partial_clears_an_utterance_without_final():
    core = make_service(FakeASR(text='xin chao anh', empty_on_finish=True))
    out = recognise(chunks(ONE), core)
    assert out and not any(r.is_final for r in out)
    assert out[0].text == 'xin chao anh?' and out[-1].text == '' and out[-1].raw_text == ''
    assert out[-1].utterance_id == 1 and out[-1].revision == out[-2].revision + 1


# ------------------------------------------------------------------ end of stream
def test_end_of_stream_chunk_audio_is_processed_then_flushed():
    out = recognise(chunks([pcm(0.4, True)]))            # the last voiced chunk carries end_of_stream
    finals = [r for r in out if r.is_final]
    assert len(finals) == 1 and finals[0].reason == pb.COMMIT_REASON_END_OF_STREAM
    assert finals[0].ts_end == 400 * 8                   # all 0.4 s were processed


def test_half_close_without_end_of_stream_flushes_too():
    out = recognise(chunks([pcm(0.4, True)], end=False))
    finals = [r for r in out if r.is_final]
    assert len(finals) == 1 and finals[0].reason == pb.COMMIT_REASON_END_OF_STREAM


def test_end_of_stream_chunk_without_audio():
    messages = chunks([pcm(0.4, True)], end=False)
    messages.append(pb.AudioChunk(session_id='call-1', stream_id='call-1-a2b', sample_rate=16000,
                                  channels=1, rtp_clock_rate=8000, chunk_seq=len(messages),
                                  language='vi', task='speech_to_text', end_of_stream=True))
    out = recognise(messages)
    assert [r.reason for r in out if r.is_final] == [pb.COMMIT_REASON_END_OF_STREAM]


def test_nothing_after_end_of_stream_is_read():
    messages = chunks(ONE)
    extra = chunks(ONE, seq_start=1000)                  # would break chunk_seq if it were read
    assert [r.is_final for r in recognise(messages + extra)][-1] is True


# ------------------------------------------------------------------ RTP timestamps
def test_ts_are_rtp_ticks_from_the_first_chunk_timestamp():
    base = recognise(chunks(ONE))
    moved = recognise(chunks(ONE, rtp_start=123456))
    assert [(r.ts_start - 123456, r.ts_end - 123456) for r in moved] == \
           [(r.ts_start, r.ts_end) for r in base]
    final = [r for r in base if r.is_final][0]
    assert final.ts_start == 0 and 1200 * 8 <= final.ts_end <= 1900 * 8


def test_ts_wrap_modulo_2_32():
    start = (1 << 32) - 4000                              # 0.5 s before the 32-bit wrap
    out = recognise(chunks(ONE, rtp_start=start))
    final = [r for r in out if r.is_final][0]
    assert final.ts_start == start
    assert 0 <= final.ts_end < (1 << 32) and final.ts_end < final.ts_start    # wrapped
    assert (final.ts_end - final.ts_start) % (1 << 32) >= 1200 * 8


def test_rtp_clock_rate_zero_means_sample_rate(caplog):
    with caplog.at_level(logging.WARNING, logger='qwen3_asr.grpc.vad'):
        out = recognise(chunks(ONE, clock=0))
    final = [r for r in out if r.is_final][0]
    assert 1200 * 16 <= final.ts_end <= 1900 * 16         # 16 ticks per ms
    assert any('rtp_clock_rate not set' in r.getMessage() for r in caplog.records)


SILENCE_VOICE_SILENCE = [pcm(0.8, False), pcm(1.2, True), pcm(0.8, False)]     # 35 chunks of 80 ms
RTP_PER_CHUNK = 640                                                            # 80 ms on an 8 kHz clock


def _final(messages):
    out = recognise(messages)
    assert out[-1].is_final
    return out[-1]


def _jump_at(index, delta):
    def mutate(messages):
        for message in messages[index:]:
            message.rtp_timestamp = (message.rtp_timestamp + delta) % (1 << 32)
    return mutate


def test_rtp_jump_reanchors_the_timeline(caplog):
    base = _final(chunks(SILENCE_VOICE_SILENCE))
    start_chunk, end_chunk = base.ts_start // RTP_PER_CHUNK, base.ts_end // RTP_PER_CHUNK
    assert 2 < start_chunk < end_chunk < 33                 # room for a jump before, inside and after

    # Jump while the utterance is open: its start keeps the old timeline, its end is on the new one.
    with caplog.at_level(logging.WARNING, logger='qwen3_asr.grpc.vad'):
        inside = _final(chunks(SILENCE_VOICE_SILENCE, mutate=_jump_at(start_chunk + 2, 80000)))
    assert (inside.ts_start, inside.ts_end) == (base.ts_start, base.ts_end + 80000)
    warnings = [r.getMessage() for r in caplog.records if 'rtp_timestamp jump' in r.getMessage()]
    assert len(warnings) == 1 and f'chunk_seq={start_chunk + 2}' in warnings[0]

    # Jump before the utterance: both ends are on the new timeline.
    before = _final(chunks(SILENCE_VOICE_SILENCE, mutate=_jump_at(2, 80000)))
    assert (before.ts_start, before.ts_end) == (base.ts_start + 80000, base.ts_end + 80000)

    # Jump after the utterance ended: its timestamps are not touched.
    after = _final(chunks(SILENCE_VOICE_SILENCE, mutate=_jump_at(end_chunk + 2, 80000)))
    assert (after.ts_start, after.ts_end) == (base.ts_start, base.ts_end)


def test_rtp_jump_backwards_and_across_the_wrap():
    base = _final(chunks(SILENCE_VOICE_SILENCE))
    back = _final(chunks(SILENCE_VOICE_SILENCE, rtp_start=500000, mutate=_jump_at(2, -400000)))
    assert (back.ts_start, back.ts_end) == (base.ts_start + 100000, base.ts_end + 100000)

    start = (1 << 32) - 1000
    wrapped = _final(chunks(SILENCE_VOICE_SILENCE, rtp_start=start, mutate=_jump_at(2, 5000)))
    assert wrapped.ts_start == (start + 5000 + base.ts_start) % (1 << 32)
    assert wrapped.ts_end == (start + 5000 + base.ts_end) % (1 << 32)


def test_position_on_a_jump_ends_on_the_old_timeline_and_starts_on_the_new_one():
    """[ts_start, ts_end) is half-open."""
    service = SpeechStreamService(make_service())
    first = chunks([pcm(0.16, True)], rtp_start=1000)[0]
    stream = service._open(first)
    for message in chunks([pcm(0.16, True)], rtp_start=1000):         # 2 chunks = 160 ms = 1280 ticks
        service._accept(stream, message)
    jumped = chunks([pcm(0.08, True)], rtp_start=50000, seq_start=2)[0]
    service._accept(stream, jumped)
    assert stream.ticks(160, end=True) == 1000 + 1280
    assert stream.ticks(160) == 50000
    assert stream.ticks(80) == 1000 + 640 and stream.ticks(200) == 50000 + 320


def test_timestamps_that_never_continue_are_logged_sparingly_and_bounded(caplog):
    """A sender whose rtp_timestamp is off on every chunk: no log flood, bounded memory."""
    def drift(messages):
        for n, message in enumerate(messages):
            message.rtp_timestamp = n * (RTP_PER_CHUNK + 1)
    with caplog.at_level(logging.WARNING, logger='qwen3_asr.grpc.vad'):
        final = _final(chunks(SILENCE_VOICE_SILENCE, mutate=drift))
    base = _final(chunks(SILENCE_VOICE_SILENCE))
    assert abs(final.ts_start - base.ts_start) < 40 and abs(final.ts_end - base.ts_end) < 40
    assert sum('rtp_timestamp jump' in r.getMessage() for r in caplog.records) == 3

    service = SpeechStreamService(make_service())
    long_call = chunks([pcm(120.0, False)], mutate=drift)               # 1500 chunks, each a jump
    stream = service._open(long_call[0])
    for message in long_call:
        service._accept(stream, message)
    assert stream.rtp_jumps == len(long_call) - 1
    assert len(stream.anchors) <= 60000 // 80 + 2                      # ANCHOR_HISTORY_MS of 80 ms chunks
    assert stream.ticks(119900) == 1498 * (RTP_PER_CHUNK + 1) + 480    # recent audio is still exact


def test_first_chunk_without_audio_does_not_count_as_a_jump(caplog):
    messages = chunks(SILENCE_VOICE_SILENCE, rtp_start=7000, seq_start=1)
    opening = pb.AudioChunk(session_id='call-1', stream_id='call-1-a2b', sample_rate=16000, channels=1,
                            rtp_timestamp=0, rtp_clock_rate=8000, chunk_seq=0, language='vi',
                            task='speech_to_text')
    with caplog.at_level(logging.WARNING, logger='qwen3_asr.grpc.vad'):
        final = _final([opening] + messages)
    base = _final(chunks(SILENCE_VOICE_SILENCE))
    assert (final.ts_start, final.ts_end) == (base.ts_start + 7000, base.ts_end + 7000)
    assert not any('rtp_timestamp jump' in r.getMessage() for r in caplog.records)


# ------------------------------------------------------------------------ chunk_seq
def test_chunk_seq_gap_is_logged_and_processing_goes_on(caplog):
    """The gateway drops chunks from its send queue when it is full; closing the stream for
    that would make it close the whole session."""
    def drop(messages):
        del messages[5:8]                                   # 3 chunks lost: seq and rtp_timestamp both jump
    base = recognise(chunks(SILENCE_VOICE_SILENCE))[-1]
    with caplog.at_level(logging.WARNING, logger='qwen3_asr.grpc.vad'):
        out = recognise(chunks(SILENCE_VOICE_SILENCE, mutate=drop))
    assert out[-1].is_final
    gaps = [r.getMessage() for r in caplog.records if 'chunk_seq gap' in r.getMessage()]
    assert len(gaps) == 1 and 'expected 5, got 8' in gaps[0]
    # The lost audio shifted nothing: the chunk after the gap re-anchored the RTP timeline.
    assert (out[-1].ts_start, out[-1].ts_end) == (base.ts_start, base.ts_end)
    assert [r.seq for r in out] == list(range(1, len(out) + 1))


def test_chunk_seq_may_start_anywhere_and_a_repeat_is_only_logged(caplog):
    assert [r.is_final for r in recognise(chunks(ONE, seq_start=41))][-1] is True

    def repeat(messages):
        messages[7].chunk_seq = messages[6].chunk_seq
    with caplog.at_level(logging.WARNING, logger='qwen3_asr.grpc.vad'):
        out = recognise(chunks(ONE, seq_start=41, mutate=repeat))
    assert out[-1].is_final
    assert sum('chunk_seq gap' in r.getMessage() for r in caplog.records) == 2     # the repeat, then the resync


def test_many_chunk_seq_gaps_are_logged_sparingly(caplog):
    def every_other(messages):
        del messages[1::2]
    with caplog.at_level(logging.WARNING, logger='qwen3_asr.grpc.vad'):
        out = recognise(chunks([pcm(2.4, True), pcm(1.6, False)], mutate=every_other))
    assert out[-1].is_final
    assert sum('chunk_seq gap' in r.getMessage() for r in caplog.records) == 3


def test_chunk_seq_minus_one_disables_the_check():
    def drop(messages):
        del messages[5]
    out = recognise(chunks(ONE, seq_start=None, mutate=drop))
    assert [r.is_final for r in out][-1] is True


# ------------------------------------------------------------------------- language
def test_language_codes_in_and_out():
    assert {r.language for r in recognise(chunks(ONE, language='en'))} == {'en'}
    assert {r.language for r in recognise(chunks(ONE, language='vi'))} == {'vi'}
    assert {r.language for r in recognise(chunks(ONE, language=''))} == {'vi'}      # server default
    assert {r.language for r in recognise(chunks(ONE, language='EN'))} == {'en'}


def test_unknown_language_code_is_logged_and_default_used(caplog):
    with caplog.at_level(logging.WARNING, logger='qwen3_asr.grpc.vad'):
        out = recognise(chunks(ONE, language='fr'))
    assert {r.language for r in out} == {'vi'} and [r.is_final for r in out][-1] is True
    assert any("unknown language code 'fr'" in r.getMessage() for r in caplog.records)


def test_language_outside_vi_en_from_the_model_is_logged_and_left_empty(caplog):
    class Detecting(FakeASR):
        def _infer(self, state):
            super()._infer(state)
            state.language = 'Chinese'

    with caplog.at_level(logging.WARNING, logger='qwen3_asr.grpc.vad'):
        out = recognise(chunks(TWO), make_service(Detecting(text='ni hao ma')))
    assert [r.raw_text for r in out if r.is_final] == ['ni hao ma', 'ni hao ma']   # still processed
    assert {r.language for r in out} == {''}
    warnings = [r for r in caplog.records if "model reported language 'Chinese'" in r.getMessage()]
    assert len(warnings) == 1                              # once per stream


def test_language_source_only_when_enabled():
    messages = chunks(ONE, language='vi', language_source='en')
    assert {r.language for r in recognise(messages)} == {'vi'}
    assert {r.language for r in recognise(messages, use_language_source=True)} == {'en'}


# ----------------------------------------------------------------------- validation
@pytest.mark.parametrize('kwargs,code', [
    (dict(task='realtime_translation'), grpc.StatusCode.UNIMPLEMENTED),
    (dict(task='keyword'), grpc.StatusCode.UNIMPLEMENTED),
    (dict(task='voice_controller_emoji'), grpc.StatusCode.UNIMPLEMENTED),
    (dict(task='noise_reduction'), grpc.StatusCode.UNIMPLEMENTED),   # gateway task, not served here yet
    (dict(task='something_else'), grpc.StatusCode.INVALID_ARGUMENT),
    (dict(task='summarise'), grpc.StatusCode.INVALID_ARGUMENT),
    (dict(session=''), grpc.StatusCode.INVALID_ARGUMENT),
    (dict(stream=''), grpc.StatusCode.INVALID_ARGUMENT),
    (dict(sample_rate=44100), grpc.StatusCode.INVALID_ARGUMENT),
    (dict(channels=2), grpc.StatusCode.INVALID_ARGUMENT),
])
def test_first_chunk_is_validated(kwargs, code):
    assert rejected(chunks(ONE, **kwargs)) == code


def test_task_may_be_empty_and_channels_unset():
    assert [r.is_final for r in recognise(chunks(ONE, task='', channels=0))][-1] is True


@pytest.mark.parametrize('field,value', [
    ('session_id', 'other-call'), ('stream_id', 'call-1-b2a'), ('sample_rate', 8000),
    ('task', ''), ('rtp_clock_rate', 16000),
])
def test_session_fields_must_not_change(field, value):
    def change(messages):
        setattr(messages[8], field, value)
    assert rejected(chunks(ONE, mutate=change)) == grpc.StatusCode.INVALID_ARGUMENT


def test_odd_pcm_length_is_rejected():
    def odd(messages):
        messages[3].pcm = messages[3].pcm[:-1]
    assert rejected(chunks(ONE, mutate=odd)) == grpc.StatusCode.INVALID_ARGUMENT


def test_results_before_a_protocol_error_are_still_delivered():
    def change(messages):
        messages[-3].sample_rate = 8000                    # late: the utterance is already committed
    service = SpeechStreamService(make_service())
    got = []

    async def run():
        async for r in service.Recognize(as_stream(chunks(TWO, mutate=change)), FakeContext()):
            got.append(r)
    with pytest.raises(Aborted):
        asyncio.run(run())
    assert [r.utterance_id for r in got if r.is_final] == [1]


# -------------------------------------------------------------------- audio formats
def test_8k_pcm_is_accepted_and_resampled():
    voiced = (np.ones(8000) * int(32767 * 0.2)).astype('<i2').tobytes()
    silence = bytes(2 * 6400)
    out = recognise(chunks([voiced, silence], sample_rate=8000, chunk_bytes=1280))
    final = [r for r in out if r.is_final][0]
    assert final.raw_text == 'xin chao anh' and 1000 * 8 <= final.ts_end <= 1700 * 8


def test_chunk_size_other_than_80ms():
    for size in (640, 3200):                               # 20 ms and 100 ms
        out = recognise(chunks(ONE, chunk_bytes=size))
        assert [r.is_final for r in out][-1] is True


# ------------------------------------------------------------------- over real gRPC
def test_real_grpc_round_trip():
    async def test(stub):
        out = [r async for r in stub.Recognize(as_stream(chunks(TWO)))]
        assert [r.utterance_id for r in out if r.is_final] == [1, 2]
        assert out[-1].text == 'xin chao anh?' and out[-1].raw_text == 'xin chao anh'
    asyncio.run(with_server(test, make_service()))


def test_real_grpc_protocol_error_status():
    async def test(stub):
        # A single message, so the client is not writing when the server aborts.
        bad = chunks([pcm(0.08, True)], task='realtime_translation')
        try:
            [r async for r in stub.Recognize(as_stream(bad))]
        except grpc.aio.AioRpcError as err:
            assert err.code() == grpc.StatusCode.UNIMPLEMENTED
            assert 'realtime_translation' in err.details()
        else:
            raise AssertionError('expected UNIMPLEMENTED')
    asyncio.run(with_server(test, make_service()))



# ------------------------------------------------------- wire compatibility with rtpgw
def _wire_fields(data):
    """Field number -> wire type of every field in a serialized message."""
    seen, pos = {}, 0
    def varint():
        nonlocal pos
        value = shift = 0
        while True:
            byte = data[pos]; pos += 1
            value |= (byte & 0x7f) << shift; shift += 7
            if byte < 0x80:
                return value
    while pos < len(data):
        key = varint()
        number, kind = key >> 3, key & 7
        seen[number] = kind
        if kind == 0:
            varint()
        elif kind == 2:
            length = varint()
            pos += length
        elif kind == 5:
            pos += 4
        elif kind == 1:
            pos += 8
        else:
            raise AssertionError(f'unexpected wire type {kind}')
    return seen


def test_result_tags_do_not_collide_with_the_gateway_topic_fields():
    """rtpgw decodes by tag: 12 = topic_label (varint), 13 = topic_info (bytes), 14 = topic_score
    (fixed32). The engine's own fields are 20-24 and it never sets 12-14."""
    fields = {f.name: f.number for f in pb.RecognitionResult.DESCRIPTOR.fields}
    assert (fields['topic_label'], fields['topic_info'], fields['topic_score']) == (12, 13, 14)
    assert [fields[n] for n in ('utterance_id', 'revision', 'raw_text', 'reason', 'status')] == [20, 21, 22, 23, 24]
    assert list(pb.TopicLabel.keys()) == ['TOPIC_UNKNOWN', 'WEATHER', 'TRAFFIC', 'SPORTS', 'FINANCE', 'NEWS',
                                           'ENTERTAINMENT', 'HEALTH', 'TECHNOLOGY']

    out = recognise(chunks(ONE))
    assert any(r.is_final for r in out) and any(not r.is_final for r in out)
    for result in out:
        on_wire = _wire_fields(result.SerializeToString())
        assert not {12, 13, 14} & set(on_wire), on_wire
        assert on_wire[20] == 0 and on_wire[21] == 0 and on_wire[22] == 2        # utterance_id, revision, raw_text
        assert set(on_wire) <= {1, 2, 3, 4, 5, 6, 8, 9, 20, 21, 22, 23}
        assert result.topic_label == pb.TOPIC_UNKNOWN and result.topic_info == '' and result.topic_score == 0


def test_gateway_audio_chunk_bytes_are_read_as_sent():
    """An AudioChunk as rtpgw's marshalAudioChunk writes it (zero values omitted, chunk_seq uint64,
    rtp_clock_rate int32)."""
    def field(number, kind, payload):
        return bytes([number << 3 | kind]) + payload
    def varint(value):
        out = bytearray()
        while True:
            out.append(value & 0x7f | (0x80 if value > 0x7f else 0)); value >>= 7
            if not value:
                return bytes(out)
    def text(number, value):
        return field(number, 2, varint(len(value)) + value.encode())
    audio = pcm(0.08, True)
    wire = (text(1, 'sess') + text(2, 'track') + field(3, 2, varint(len(audio)) + audio) + field(4, 0, varint(16000))
            + field(5, 0, varint(1)) + field(6, 0, varint(4294967000)) + field(7, 0, varint(80))
            + text(9, 'vi') + text(10, 'speech_to_text') + field(11, 0, varint(8000)) + field(12, 0, varint(1800)))
    chunk = pb.AudioChunk.FromString(wire)
    assert (chunk.session_id, chunk.stream_id, chunk.sample_rate, chunk.channels) == ('sess', 'track', 16000, 1)
    assert (chunk.rtp_timestamp, chunk.duration_ms, chunk.rtp_clock_rate, chunk.chunk_seq) == (4294967000, 80, 8000, 1800)
    assert (chunk.language, chunk.task, chunk.end_of_stream, chunk.language_source) == ('vi', 'speech_to_text', False, '')
    assert chunk.pcm == audio
