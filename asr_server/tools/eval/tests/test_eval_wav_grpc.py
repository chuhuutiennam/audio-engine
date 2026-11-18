import asyncio
from pathlib import Path
import sys

import grpc
import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import speech_pb2 as pb
import speech_pb2_grpc as pb_grpc
from eval_wav_grpc import build_parser, load_wav_as_pcm16, run_eval, normalize_wer, wer_details


@pytest.mark.parametrize('reference,hypothesis,expected', [
    ('Xin chào bạn', 'xin chào anh', (1, 0, 0, 1/3)),
    ('xin chào bạn', 'xin bạn', (0, 1, 0, 1/3)),
    ('xin chào', 'xin em chào', (0, 0, 1, 1/2)),
    ('Xin chào, thế giới!', 'xin chào thế giới', (0, 0, 0, 0.0)),
    ('mã thuê bao 123', 'mã thue bao 123', (1, 0, 0, 1/4)),
])
def test_wer(reference, hypothesis, expected):
    score = wer_details(reference, hypothesis)
    assert (score['substitutions'], score['deletions'], score['insertions']) == expected[:3]
    assert score['wer'] == pytest.approx(expected[3])


def test_unicode_nfc_and_punctuation():
    assert normalize_wer('HOÀNG! Anh,Em?') == 'hoàng anh em'
    assert normalize_wer('Hòa') == normalize_wer('Hòa')


def test_ref_empty_is_error():
    with pytest.raises(ValueError, match='no words'):
        wer_details('...!', 'xin chao')


def test_read_stereo_48k_and_resample_16k(tmp_path):
    sr = 48000
    ts = np.arange(sr, dtype=np.float32) / sr
    left = (0.3 * np.sin(2*np.pi*300*ts)).astype(np.float32)
    right = (0.3 * np.sin(2*np.pi*300*ts)).astype(np.float32)
    wav = tmp_path / 'voice.wav'
    sf.write(str(wav), np.stack([left, right], axis=1), sr, subtype='PCM_16')
    payload, info = load_wav_as_pcm16(wav, None)
    assert info['original_channels'] == 2
    assert info['original_sample_rate'] == 48000
    assert info['sample_rate'] == 16000
    # 8 kHz input is upsampled too: the gateway always delivers 16 kHz
    wav8 = tmp_path / 'voice8.wav'
    sf.write(str(wav8), left[::6], 8000, subtype='PCM_16')
    assert load_wav_as_pcm16(wav8, None)[1]['sample_rate'] == 16000
    assert load_wav_as_pcm16(wav8, 8000)[1]['sample_rate'] == 8000
    assert len(payload) == 16000 * 2
    assert info['duration_sec'] == pytest.approx(1.0)


class MockSpeechServer(pb_grpc.SpeechStreamServicer):
    def __init__(self):
        self.chunks = []
        self.ended = False

    async def Recognize(self, request_iterator, context):
        async for chunk in request_iterator:
            self.chunks.append(chunk)
            ids = dict(session_id=chunk.session_id, stream_id=chunk.stream_id)
            if chunk.chunk_seq == 0:
                yield pb.RecognitionResult(**ids, seq=1, utterance_id=1, revision=1,
                                           text='xin', raw_text='xin', is_final=False)
            elif chunk.chunk_seq == 1:
                yield pb.RecognitionResult(**ids, seq=2, utterance_id=1, revision=2,
                                           text='xin chào', raw_text='xin chào', is_final=True,
                                           reason=pb.COMMIT_REASON_SILENCE, language='vi',
                                           ts_start=5000, ts_end=5000 + 160 * 8)
            if chunk.end_of_stream:
                self.ended = True
                yield pb.RecognitionResult(**ids, seq=3, utterance_id=2, revision=1,
                                           text='thế giới.', raw_text='thế giới', is_final=True,
                                           reason=pb.COMMIT_REASON_END_OF_STREAM, language='vi',
                                           ts_start=5000 + 160 * 8, ts_end=5000 + 250 * 8)


def test_full_grpc_client_sends_80ms_and_wer_zero(tmp_path):
    async def test():
        sr = 8000
        # Exactly 250ms input = 80+80+80+10ms => 640,640,640,80 samples.
        wav = tmp_path / 'audio.wav'
        sf.write(str(wav), np.zeros(2000, dtype=np.int16), sr, subtype='PCM_16')
        truth = tmp_path / 'ground_truth.txt'
        truth.write_text('Xin chào thế giới!\n', encoding='utf-8')
        fake = MockSpeechServer()
        server = grpc.aio.server()
        pb_grpc.add_SpeechStreamServicer_to_server(fake, server)
        port = server.add_insecure_port('127.0.0.1:0')
        await server.start()
        try:
            args = build_parser().parse_args([
                '--wav', str(wav), '--truth', str(truth),
                '--target', f'127.0.0.1:{port}',
                '--output-dir', str(tmp_path / 'results'),
                '--fast', '--quiet-partials', '--rate', '8000', '--rtp-start', '5000',
                '--call-id', 'call-9', '--leg-id', 'call-9-a2b',
            ])
            results = await run_eval(args)
            assert fake.ended
            sent = fake.chunks
            assert [c.chunk_seq for c in sent] == [0, 1, 2, 3]                 # 0-based
            assert [len(c.pcm) for c in sent] == [1280, 1280, 1280, 160]
            assert [c.end_of_stream for c in sent] == [False, False, False, True]
            assert [c.rtp_timestamp for c in sent] == [5000, 5640, 6280, 6920]  # +640 ticks per 80 ms
            assert [c.duration_ms for c in sent] == [80, 80, 80, 10]
            for c in sent:
                assert (c.session_id, c.stream_id) == ('call-9', 'call-9-a2b')
                assert (c.sample_rate, c.channels, c.rtp_clock_rate) == (8000, 1, 8000)
                assert (c.language, c.task) == ('vi', 'speech_to_text')
            assert results['asr_text'] == 'xin chào thế giới.'
            assert results['metrics']['wer'] == 0.0
            assert results['committed_utterances'] == 2
            assert results['response_count'] == 3
            assert (tmp_path / 'results/asr.txt').read_text(encoding='utf-8').strip() == 'xin chào thế giới.'
            assert 'wer_percent' in (tmp_path / 'results/metrics.json').read_text(encoding='utf-8')
            lines = (tmp_path / 'results/stream_events.jsonl').read_text(encoding='utf-8').splitlines()
            assert len(lines) == 3
            import json
            commit = json.loads(lines[1])
            assert commit['reason'] == 'COMMIT_REASON_SILENCE' and commit['utterance_id'] == 1
            assert (commit['ts_start'], commit['ts_end']) == (5000, 6280)
            assert (commit['start_ms'], commit['end_ms']) == (0.0, 160.0)   # relative to --rtp-start
        finally:
            await server.stop(grace=0)
    asyncio.run(test())
