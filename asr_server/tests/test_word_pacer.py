"""Word-by-word reveal of PARTIAL results (word_reveal.* in config/server.yaml)."""
import asyncio
import time

import grpc
import pytest

from asr_server import messages as m
from asr_server.proto import speech_pb2_grpc
from asr_server.speech_service import SpeechStreamService
from asr_server.word_pacer import WordPacer, paced_transcripts

from test_regression import FakeASR, FakeContext, make_service, pcm, speech_req
from test_speech_grpc import ONE, as_stream, chunks, recognise

WINDOW = 400
MIN_GAP = 120


def partial(text, utterance=1, raw=None, end_ms=1000):
    return m.Transcript(call_id='c', leg_id='l', utterance_index=utterance, text=text,
                        raw_text=text if raw is None else raw, is_final=False, revision=99,
                        language='Vietnamese', reason='partial', event=m.PARTIAL,
                        start_ms=100, end_ms=end_ms)


def final(text, utterance=1, end_ms=2000):
    return m.Transcript(call_id='c', leg_id='l', utterance_index=utterance, text=text, raw_text=text,
                        is_final=True, revision=99, language='Vietnamese', reason='vad_endpoint',
                        event=m.COMMIT, commit_id=1, start_ms=100, end_ms=end_ms)


def pacer():
    return WordPacer(window_ms=WINDOW, min_interval_ms=MIN_GAP)


def drain(p):
    """Run every scheduled tick; returns [(time_ms, transcript), ...]."""
    out = []
    while p.due_ms() is not None:
        now = p.due_ms()
        out.extend((now, t) for t in p.tick(now))
    return out


# --------------------------------------------------------------------- pacing
def test_first_word_is_shown_at_once_and_the_rest_within_the_window():
    p = pacer()
    assert [t.text for t in p.push(partial('mot hai ba bon'), 0)] == ['mot']
    timed = drain(p)
    assert [t.text for _, t in timed] == ['mot hai', 'mot hai ba', 'mot hai ba bon']
    assert timed[-1][0] <= WINDOW


def test_two_sends_are_never_closer_than_the_minimum_gap():
    p = pacer()
    p.push(partial('mot hai ba bon nam'), 0)
    times = [0] + [when for when, _ in drain(p)]
    assert all(b - a >= MIN_GAP for a, b in zip(times, times[1:]))


def test_many_words_are_grouped_so_the_text_does_not_fall_behind():
    p = pacer()
    words = [f'w{i}' for i in range(13)]
    p.push(partial(' '.join(words)), 0)
    timed = drain(p)
    assert timed[-1][1].text == ' '.join(words)
    assert timed[-1][0] <= WINDOW
    assert len(timed) <= WINDOW // MIN_GAP


def test_a_new_partial_during_the_reveal_never_shrinks_the_text():
    p = pacer()
    shown = [t.text for t in p.push(partial('mot hai ba bon'), 0)]
    first_tick = p.due_ms()
    shown += [t.text for t in p.tick(first_tick)]
    shown += [t.text for t in p.push(partial('mot hai ba bon nam sau'), first_tick + 130)]
    timed = drain(p)
    shown += [t.text for _, t in timed]
    counts = [len(text.split()) for text in shown]
    assert counts == sorted(counts) and shown[-1] == 'mot hai ba bon nam sau'
    assert timed[-1][0] <= first_tick + 130 + WINDOW


def test_unchanged_text_sends_nothing():
    p = pacer()
    p.push(partial('mot'), 0)
    assert p.push(partial('mot'), 500) == [] and p.due_ms() is None


# ----------------------------------------------------------------- corrections
def test_a_corrected_word_is_replaced_at_once_without_shrinking():
    p = pacer()
    p.push(partial('toi di hoc'), 0)
    drain(p)
    assert [t.text for t in p.push(partial('toi da hoc bai roi'), 1000)] == ['toi da hoc']
    assert [t.text for _, t in drain(p)] == ['toi da hoc bai', 'toi da hoc bai roi']


def test_a_correction_right_after_a_send_waits_for_the_minimum_gap():
    p = pacer()
    p.push(partial('toi'), 0)
    assert p.push(partial('ban'), 50) == []
    assert p.due_ms() == MIN_GAP
    assert [t.text for t in p.tick(MIN_GAP)] == ['ban']


def test_a_shorter_text_is_cut_at_once():
    p = pacer()
    p.push(partial('mot hai ba'), 0)
    drain(p)
    assert [t.text for t in p.push(partial('mot hai'), 1000)] == ['mot hai']
    assert p.due_ms() is None


# ------------------------------------------------------------- final and clear
def test_final_is_immediate_and_cancels_the_pending_words():
    p = pacer()
    p.push(partial('mot hai ba bon'), 0)
    out = p.push(final('mot hai ba bon.'), 10)
    assert [(t.text, t.is_final) for t in out] == [('mot hai ba bon.', True)]
    assert p.due_ms() is None and p.tick(10_000) == []


def test_empty_partial_is_immediate_and_cancels_the_pending_words():
    p = pacer()
    p.push(partial('mot hai ba bon'), 0)
    out = p.push(partial(''), 10)
    assert [t.text for t in out] == [''] and p.due_ms() is None


def test_flush_sends_the_words_still_waiting():
    p = pacer()
    p.push(partial('mot hai ba'), 0)
    assert [t.text for t in p.flush()] == ['mot hai ba']
    assert p.flush() == [] and p.due_ms() is None


# ------------------------------------------------------------------- contract
def test_revisions_increase_by_one_and_restart_after_a_final():
    p = pacer()
    out = p.push(partial('mot hai ba'), 0)
    out += [t for _, t in drain(p)]
    out += p.push(final('mot hai ba.'), 1000)
    assert [t.revision for t in out] == list(range(1, len(out) + 1))
    assert p.push(partial('bon', utterance=2), 2000)[0].revision == 1


def test_revisions_keep_growing_when_an_utterance_index_is_reused():
    p = pacer()
    first = p.push(partial('mot'), 0)[0].revision
    cleared = p.push(partial(''), 500)[0].revision
    again = p.push(partial('hai'), 1000)[0].revision
    assert first < cleared < again


def test_text_and_raw_text_stay_aligned_word_by_word():
    p = pacer()
    source = partial('xin chao, cac ban.', raw='xin chao cac ban')
    out = p.push(source, 0) + [t for _, t in drain(p)]
    assert [(t.text, t.raw_text) for t in out] == [
        ('xin', 'xin'), ('xin chao,', 'xin chao'), ('xin chao, cac', 'xin chao cac'),
        ('xin chao, cac ban.', 'xin chao cac ban')]


def test_text_that_cannot_be_aligned_with_raw_text_is_sent_whole():
    p = pacer()
    out = p.push(partial('mot hai ba', raw='mot hai'), 0)
    assert [t.text for t in out] == ['mot hai ba'] and p.due_ms() is None


def test_other_fields_are_kept_and_the_source_is_not_modified():
    p = pacer()
    source = partial('mot hai', end_ms=1234)
    sent = p.push(source, 0)[0]
    assert (sent.start_ms, sent.end_ms, sent.language, sent.reason) == (100, 1234, 'Vietnamese', 'partial')
    assert source.text == 'mot hai' and source.revision == 99


def test_a_new_raw_text_is_sent_even_when_the_formatted_text_is_the_same():
    p = pacer()
    p.push(partial('chao.', raw='chao'), 0)
    assert [t.raw_text for t in p.push(partial('chao.', raw='chao.'), 500)] == ['chao.']


def test_the_first_word_is_shown_alone_even_right_after_a_final():
    p = pacer()
    p.push(final('cau mot'), 10)
    assert p.push(partial('a b c d', utterance=2), 50) == []
    timed = drain(p)
    assert timed[0][1].text == 'a' and timed[-1][1].text == 'a b c d'
    assert timed[-1][0] <= 50 + WINDOW


def test_text_that_cannot_be_aligned_still_respects_the_minimum_gap():
    p = pacer()
    p.push(partial('mot'), 0)
    assert p.push(partial('mot hai ba', raw='mot hai'), 10) == []
    assert p.due_ms() == MIN_GAP
    assert [t.text for t in p.tick(MIN_GAP)] == ['mot hai ba']


@pytest.mark.parametrize('window, gap', [(0, 0), (100, 0), (100, 200), (-1, 10)])
def test_invalid_timing_is_rejected(window, gap):
    with pytest.raises(ValueError):
        WordPacer(window_ms=window, min_interval_ms=gap)


# --------------------------------------------------------------- async wrapper
def collect(source, *, window=120, gap=30):
    async def run():
        start = time.monotonic()
        return [(time.monotonic() - start, t)
                async for t in paced_transcripts(source, WordPacer(window_ms=window, min_interval_ms=gap))]
    return asyncio.run(run())


def test_words_keep_coming_while_the_model_is_busy():
    async def source():
        yield partial('mot hai ba bon')
        await asyncio.sleep(0.5)                # the core is waiting for the model
        yield final('mot hai ba bon.')

    out = collect(source())
    partials = [(when, t.text) for when, t in out if not t.is_final]
    assert [text for _, text in partials] == ['mot', 'mot hai', 'mot hai ba', 'mot hai ba bon']
    assert partials[-1][0] < 0.4 and out[-1][1].is_final


def test_stream_end_sends_the_words_still_waiting():
    async def source():
        yield partial('mot hai ba bon')

    assert collect(source(), window=5000, gap=1000)[-1][1].text == 'mot hai ba bon'


def test_an_error_of_the_core_reaches_the_caller():
    class Failure(Exception):
        pass

    async def source():
        yield partial('mot')
        raise Failure('model failed')

    with pytest.raises(Failure):
        collect(source())


def test_a_cancelled_core_ends_the_stream():
    async def source():
        yield partial('mot')
        raise asyncio.CancelledError()

    async def run():
        stream = paced_transcripts(source(), pacer())
        await stream.__anext__()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(stream.__anext__(), 2)
    asyncio.run(run())


def test_closing_the_core_ends_every_open_stream():
    async def run():
        core = make_service(FakeASR(text='mot hai ba', delay=0.3))
        service = SpeechStreamService(core, word_reveal=(WINDOW, MIN_GAP))

        async def one(n):
            messages = chunks(ONE, session=f's{n}', stream=f's{n}')
            async for _ in service.Recognize(as_stream(messages), FakeContext()):
                pass

        tasks = [asyncio.ensure_future(one(n)) for n in range(3)]
        await asyncio.sleep(0.5)
        core.close()
        done, pending = await asyncio.wait(tasks, timeout=5)
        for task in done:
            task.cancelled() or task.exception()
        for task in pending:
            task.cancel()
        return len(pending)

    assert asyncio.run(run()) == 0


def test_closing_the_stream_stops_the_core():
    closed = asyncio.Event

    async def run():
        stopped = closed()

        async def source():
            try:
                yield partial('mot hai')
                await asyncio.sleep(30)
            finally:
                stopped.set()

        stream = paced_transcripts(source(), pacer())
        await stream.__anext__()
        await stream.aclose()
        await asyncio.wait_for(stopped.wait(), 1)
    asyncio.run(run())


def test_a_slow_client_slows_the_core_down():
    produced = 0

    async def source():
        nonlocal produced
        for n in range(200):
            produced += 1
            yield final(f'cau {n}', utterance=n + 1)

    async def run():
        stream = paced_transcripts(source(), pacer())
        await stream.__anext__()
        await asyncio.sleep(0.05)
        await stream.aclose()
    asyncio.run(run())
    assert produced < 50


# ----------------------------------------------------------------- gRPC service
def test_final_is_the_same_with_and_without_the_reveal():
    plain = recognise(chunks(ONE), make_service())
    paced = recognise(chunks(ONE), make_service(), word_reveal=(WINDOW, MIN_GAP))
    fields = lambda r: (r.text, r.raw_text, r.ts_start, r.ts_end, r.reason, r.utterance_id)
    assert [fields(r) for r in paced if r.is_final] == [fields(r) for r in plain if r.is_final]
    assert [r.seq for r in paced] == list(range(1, len(paced) + 1))
    assert [r.revision for r in paced] == list(range(1, len(paced) + 1))


def test_reveal_is_off_by_default():
    assert SpeechStreamService(make_service()).word_reveal is None


def test_reveal_through_a_real_grpc_call():
    async def run():
        server = grpc.aio.server()
        core = make_service(FakeASR(text='mot hai ba bon', delay=0.25), chunk_sec=0.4)
        speech_pb2_grpc.add_SpeechStreamServicer_to_server(
            SpeechStreamService(core, word_reveal=(120, 30)), server)
        port = server.add_insecure_port('127.0.0.1:0')
        await server.start()
        try:
            async with grpc.aio.insecure_channel(f'127.0.0.1:{port}') as channel:
                call = speech_pb2_grpc.SpeechStreamStub(channel).Recognize(
                    speech_req([pcm(1.2, True), pcm(0.8, False)]))
                return [r async for r in call]
        finally:
            await server.stop(0)
            core.close()

    out = asyncio.run(run())
    partials = [r.raw_text for r in out if not r.is_final]
    assert partials[:4] == ['mot', 'mot hai', 'mot hai ba', 'mot hai ba bon']
    assert out[-1].is_final and out[-1].raw_text == 'mot hai ba bon'
    assert [r.revision for r in out] == list(range(1, len(out) + 1))


def test_a_rejected_call_keeps_its_grpc_status():
    async def run():
        server = grpc.aio.server()
        core = make_service()
        speech_pb2_grpc.add_SpeechStreamServicer_to_server(
            SpeechStreamService(core, word_reveal=(WINDOW, MIN_GAP)), server)
        port = server.add_insecure_port('127.0.0.1:0')
        await server.start()

        async def bad_requests():
            yield chunks(ONE, sample_rate=44100, clock=44100)[0]

        try:
            async with grpc.aio.insecure_channel(f'127.0.0.1:{port}') as channel:
                call = speech_pb2_grpc.SpeechStreamStub(channel).Recognize(bad_requests())
                with pytest.raises(grpc.aio.AioRpcError) as caught:
                    async for _ in call:
                        pass
                return caught.value.code()
        finally:
            await server.stop(0)
            core.close()

    assert asyncio.run(run()) == grpc.StatusCode.INVALID_ARGUMENT
