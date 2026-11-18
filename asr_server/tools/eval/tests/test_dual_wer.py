import asyncio
import json
from pathlib import Path
import sys

import grpc
import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import speech_pb2 as pb
import speech_pb2_grpc as pb_grpc
from eval_wav_grpc import run_eval, build_parser


class FinalOnlyServer(pb_grpc.SpeechStreamServicer):
    def __init__(self, raw, formatted):
        self.raw = raw
        self.formatted = formatted

    async def Recognize(self, request_iterator, context):
        async for chunk in request_iterator:
            if chunk.end_of_stream:
                yield pb.RecognitionResult(
                    session_id=chunk.session_id,
                    stream_id=chunk.stream_id,
                    utterance_id=1,
                    seq=1,
                    is_final=True,
                    raw_text=self.raw,
                    text=self.formatted,
                    reason=pb.COMMIT_REASON_END_OF_STREAM,
                )


async def do_case(tmp_path, raw, formatted, truth='Xin chào anh'):
    wav = tmp_path / 'wav.wav'
    sf.write(str(wav), np.zeros(640, dtype='<i2'), 8000, subtype='PCM_16')
    gt = tmp_path / 'truth.txt'
    gt.write_text(truth, encoding='utf-8')
    server = grpc.aio.server()
    pb_grpc.add_SpeechStreamServicer_to_server(FinalOnlyServer(raw, formatted), server)
    port = server.add_insecure_port('127.0.0.1:0')
    await server.start()
    try:
        output = tmp_path / 'results'
        args = build_parser().parse_args([
            '--wav', str(wav), '--truth', str(gt), '--fast',
            '--target', f'127.0.0.1:{port}', '--output-dir', str(output),
            '--vram-mode', 'off',
        ])
        result = await run_eval(args)
        return result, output
    finally:
        await server.stop(0)


def test_raw_formatted_are_separate_but_equal_wer_when_only_punctuation(tmp_path):
    async def go():
        result, out = await do_case(
            tmp_path,
            'language: Vietnamese<asr_text>Xin chào em',
            'language: Vietnamese<asr_text>Xin, chào em.',
        )
        assert result['asr_before_punctuation'] == 'Xin chào em'
        assert result['asr_after_punctuation'] == 'Xin, chào em.'
        assert result['wer_before_punctuation']['wer_percent'] == pytest.approx(33.333)
        assert result['wer_after_punctuation']['wer_percent'] == pytest.approx(33.333)
        assert result['lexical_content_changed_by_punctuation'] is False
        assert (out / 'asr_before_punctuation.txt').read_text(encoding='utf-8').strip() == 'Xin chào em'
        assert (out / 'asr_after_punctuation.txt').read_text(encoding='utf-8').strip() == 'Xin, chào em.'
        assert (out / 'asr.txt').read_text(encoding='utf-8').strip() == 'Xin, chào em.'
        m = json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
        assert m['wer_before_punctuation']['wer_percent'] == pytest.approx(33.333)
        assert m['wer_after_punctuation']['wer_percent'] == pytest.approx(33.333)
        events = [json.loads(s) for s in (out / 'stream_events.jsonl').read_text(encoding='utf-8').splitlines()]
        assert events[0]['asr_before_punctuation'] == 'Xin chào em'
        assert events[0]['asr_after_punctuation'] == 'Xin, chào em.'
        assert 'language:' in events[0]['raw_text_original']
    asyncio.run(go())


def test_detect_lexical_change_in_punctuation_stage(tmp_path):
    async def go():
        result, _ = await do_case(tmp_path, 'Xin chào em', 'Xin, chào anh.')
        assert result['wer_before_punctuation']['wer_percent'] == pytest.approx(33.333)
        assert result['wer_after_punctuation']['wer_percent'] == pytest.approx(0)
        assert result['lexical_content_changed_by_punctuation'] is True
    asyncio.run(go())


def test_missing_raw_text_reports_no_raw_wer_not_fake_score(tmp_path):
    async def go():
        result, _ = await do_case(tmp_path, '', 'Xin, chào anh.')
        assert result['asr_before_punctuation'] == ''
        assert result['wer_before_punctuation'] is None
        assert result['raw_text_available_for_all_commits'] is False
        assert result['missing_raw_utterance_indices'] == [1]
        assert result['wer_after_punctuation']['wer_percent'] == 0
        assert result['metrics']['wer_percent'] == 0
    asyncio.run(go())
