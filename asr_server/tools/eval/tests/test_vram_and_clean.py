import asyncio
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from asr_text_clean import strip_asr_markers
from eval_wav_grpc import wer_details
from vram_monitor import GpuVramMonitor
from apply_server_text_fix import patch


def test_remove_repeated_qwen_prefix_even_adjacent_to_words():
    orig = ('language: Vietnamese<asr_text>alo chào anh ' 
            'language: Vietnamese<asr_text>mình tên là Phạm Văn Hùng. '
            'language Vietnamese<asr_text>số điện thoại 0901234567?')
    assert strip_asr_markers(orig) == ('alo chào anh mình tên là Phạm Văn Hùng. '
                                      'số điện thoại 0901234567?')


def test_repeated_prefix_after_word_without_separator():
    assert strip_asr_markers('xinlanguage: Vietnamese<asr_text>chào') == 'xin chào'
    assert strip_asr_markers('hello <|asr_text|> world') == 'hello world'
    assert strip_asr_markers('câu này không cần sửa') == 'câu này không cần sửa'


def test_wer_ignores_protocol_metadata_only_in_hypothesis():
    assert wer_details('Alo chào anh', 'language: Vietnamese<asr_text>Alo chào anh')['wer']==0.0


class FakeNVMLBackend:
    def __init__(self):
        self.i=0;self.closed=False
        self.series=[2048,2112,2304,2176]
    def read(self):
        used=self.series[min(self.i,len(self.series)-1)]
        self.i+=1
        return [{'gpu_index':0,'name':'Test GPU', 'used_mib':float(used), 'total_mib':8192.0}]
    def close(self):
        self.closed=True


def test_vram_sampler_baseline_peak_after(tmp_path):
    async def test():
        nvml=FakeNVMLBackend()
        sampler=GpuVramMonitor(nvml,interval_ms=20)
        await sampler.start()
        await asyncio.sleep(.065)
        await sampler.stop()
        sampler.save_csv(tmp_path/'vram_samples.csv')
        r=sampler.summary()
        assert nvml.closed
        assert r['status']=='ok'
        gpu=r['per_gpu'][0]
        assert gpu['baseline_used_mib']==2048.0
        assert gpu['peak_used_mib']==2304.0
        assert gpu['peak_delta_mib']==256.0
        assert gpu['end_delta_mib']==128.0
        assert len((tmp_path/'vram_samples.csv').read_text().splitlines())>=4
    asyncio.run(test())


def test_server_patch_creates_backup_and_keeps_raw_decoded(tmp_path):
    content = '''from vad import WebRTCEndpointer\nnew_text = asr_state.text or ''\nraw = asr_state.text or ''\nstate._raw_decoded = 'prefix'\n'''
    p=tmp_path/'server.py'
    p.write_text(content)
    backup=patch(p)
    edited=p.read_text()
    assert backup.read_text()==content
    assert 'new_text = strip_asr_markers(asr_state.text or "")' in edited
    assert 'raw = strip_asr_markers(asr_state.text or "")' in edited
    assert 'state._raw_decoded' in edited
    assert (tmp_path/'asr_text_clean.py').exists()


def test_server_patch_fails_safe_unknown_version(tmp_path):
    p=tmp_path/'server.py'
    p.write_text('something else')
    with pytest.raises(ValueError):
        patch(p)
    assert p.read_text()=='something else'


def test_full_grpc_mock_cleans_wrappers_before_wer(tmp_path):
    import grpc
    import numpy as np
    import soundfile as sf
    import speech_pb2 as pb
    import speech_pb2_grpc as pb_grpc
    from eval_wav_grpc import run_eval,build_parser

    class Stub(pb_grpc.SpeechStreamServicer):
        async def Recognize(self, requests, context):
            async for chunk in requests:
                if chunk.end_of_stream:
                    text='language: Vietnamese<asr_text>Xin chào ' \
                         'language Vietnamese<asr_text>anh.'
                    yield pb.RecognitionResult(session_id=chunk.session_id,stream_id=chunk.stream_id,
                                               text=text,raw_text=text,is_final=True,
                                               seq=1,utterance_id=1)

    async def test():
        wav=tmp_path/'a.wav'
        truth=tmp_path/'truth.txt'
        sf.write(wav, np.zeros(800,dtype=np.int16), 8000, subtype='PCM_16')
        truth.write_text('Xin chào anh', encoding='utf-8')
        server=grpc.aio.server()
        pb_grpc.add_SpeechStreamServicer_to_server(Stub(),server)
        port=server.add_insecure_port('127.0.0.1:0')
        await server.start()
        try:
            args=build_parser().parse_args(['--wav',str(wav),'--truth',str(truth),
                  '--target',f'127.0.0.1:{port}','--vram-mode','off','--fast',
                  '--output-dir',str(tmp_path/'out')])
            result=await run_eval(args)
            assert result['asr_text']=='Xin chào anh.'
            assert result['metrics']['wer']==0
            events=[__import__('json').loads(s) for s in (tmp_path/'out'/'stream_events.jsonl').read_text().splitlines()]
            assert events[0]['text']=='Xin chào anh.'
            assert 'language' in events[0]['text_original']
        finally:
            await server.stop(0)
    asyncio.run(test())
