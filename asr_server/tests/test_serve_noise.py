"""server.serve() with the task switches asr.enabled and noise_reduction.enabled.

Real gRPC server on a free local port; the models are stand-ins (FakeDenoiser of test_denoise.py,
the fake ASR core of test_regression.py). tests/test_noise_real_model.py runs DPDFNet-2 itself.
"""
import asyncio
import contextlib
import importlib.abc
import logging
import socket
import sys
import types

import grpc
import pytest

from asr_server import asr_backend, denoise, server
from asr_server.config import Settings
from asr_server.noise_service import NoiseReductionService
from asr_server.proto import speech_pb2_grpc

from test_denoise import FakeDenoiser
from test_noise_grpc import DATA, SlowDenoiser, noise_chunks
from test_regression import make_service
from test_speech_grpc import ONE, as_stream, chunks


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture(autouse=True)
def offline_env(monkeypatch):
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE',
                 'VLLM_WORKER_MULTIPROC_METHOD'):
        monkeypatch.setenv(name, '1')


@pytest.fixture
def fake_denoisers(monkeypatch):
    """dpdfnet2_factory -> FakeDenoiser; records the arguments and every denoiser made."""
    calls = {'args': [], 'made': []}

    def factory(model, num_threads):
        calls['args'].append((model, num_threads))

        def make():
            calls['made'].append(FakeDenoiser())
            return calls['made'][-1]
        return make
    monkeypatch.setattr(denoise, 'dpdfnet2_factory', factory)
    return calls


@pytest.fixture
def no_asr(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError('the ASR models must not be loaded')
    monkeypatch.setattr(server, 'ASRService', refuse)
    monkeypatch.setattr(server, 'HFPunctuator', refuse)
    monkeypatch.setattr(asr_backend, 'VLLMASRBackend', refuse)


@pytest.fixture
def fake_asr(monkeypatch):
    monkeypatch.setattr(server, 'ASRService', lambda *a, **k: make_service())
    monkeypatch.setattr(server, 'HFPunctuator', lambda *a, **k: object())
    monkeypatch.setattr(asr_backend, 'VLLMASRBackend', lambda **k: object())


def settings(tmp_path, **values):
    model = tmp_path / 'dpdfnet2.onnx'
    model.write_bytes(b'onnx')
    # noise_max_backlog_ms 0: the tests send their audio at once, which the limit is meant to refuse
    base = dict(host='127.0.0.1', port=free_port(), model=str(tmp_path), punct_model=str(tmp_path),
                vad_backend='webrtc', noise_model=str(model), noise_max_streams=2,
                noise_max_backlog_ms=0)
    base.update(values)
    return Settings(**base)


async def with_running_server(s, check):
    task = asyncio.create_task(server.serve(s))
    try:
        async with grpc.aio.insecure_channel(f'127.0.0.1:{s.port}') as channel:
            ready = asyncio.create_task(channel.channel_ready())
            await asyncio.wait({task, ready}, timeout=20, return_when=asyncio.FIRST_COMPLETED)
            if task.done():
                task.result()                       # serve() failed: raise its error
            assert ready.done(), 'server did not start'
            # Bounded: a full-suite run once hung in a test of this file (not reproduced since);
            # a hang now fails with the place it happened.
            await asyncio.wait_for(check(speech_pb2_grpc.SpeechStreamStub(channel)), 60)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, 30)        # server.stop() itself waits at most 5 s


async def status_of(stub, messages):
    try:
        [r async for r in stub.Recognize(as_stream(messages))]
    except grpc.aio.AioRpcError as err:
        return err.code()
    return grpc.StatusCode.OK


async def noise_ok(stub):
    out = [r async for r in stub.Recognize(as_stream(noise_chunks()))]
    assert out[-1].is_final and sum(len(r.audio_payload) for r in out) == len(DATA)


def test_noise_only_server_loads_no_asr_and_serves_noise_reduction(tmp_path, fake_denoisers, no_asr):
    s = settings(tmp_path, asr_enabled=False, noise_enabled=True, model='/missing/Qwen3-ASR',
                 punct_model='/missing/punct', noise_num_threads=3)

    async def check(stub):
        await noise_ok(stub)
        # one chunk only: see test_noise_grpc.test_over_grpc_results_and_client_cancel
        assert await status_of(stub, chunks(ONE)[:1]) == grpc.StatusCode.UNIMPLEMENTED
    asyncio.run(with_running_server(s, check))
    assert fake_denoisers['args'] == [(s.noise_model, 3)]
    assert len(fake_denoisers['made']) == 2                  # noise_reduction.max_streams, loaded at start


def test_both_tasks_on_one_server(tmp_path, fake_denoisers, fake_asr):
    s = settings(tmp_path, noise_enabled=True)

    async def check(stub):
        await noise_ok(stub)
        stt = [r async for r in stub.Recognize(as_stream(chunks(ONE)))]
        assert stt[-1].is_final and stt[-1].text
    asyncio.run(with_running_server(s, check))


def test_serve_applies_the_configured_backlog_limit(tmp_path, no_asr, monkeypatch):
    # review L4: max_backlog_ms of server.yaml was not passed on (every stream got 500).
    # The model must be slower than the audio: over gRPC the chunks arrive one by one, and a
    # fast model takes each one before the next arrives, so no backlog builds up.
    monkeypatch.setattr(denoise, 'dpdfnet2_factory', lambda model, num_threads: SlowDenoiser)
    s = settings(tmp_path, asr_enabled=False, noise_enabled=True, noise_max_backlog_ms=200)

    async def check(stub):
        # 5 chunks at once; the first takes 200 ms in the model, so 3 chunks (240 ms) wait when
        # the second is taken. Few enough that the client has sent them all before the RPC ends
        # (a client still writing would see INTERNAL).
        try:
            [r async for r in stub.Recognize(as_stream(noise_chunks()[:5]))]
        except grpc.aio.AioRpcError as err:
            assert err.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
            assert 'max_backlog_ms 200' in err.details()
            return
        raise AssertionError('expected RESOURCE_EXHAUSTED')
    asyncio.run(with_running_server(s, check))


def test_noise_reduction_off_by_default_loads_no_denoiser(tmp_path, fake_denoisers, fake_asr):
    s = settings(tmp_path)

    async def check(stub):
        assert await status_of(stub, noise_chunks()[:1]) == grpc.StatusCode.UNIMPLEMENTED
    asyncio.run(with_running_server(s, check))
    assert fake_denoisers['args'] == []


def test_noise_service_is_closed_when_the_server_stops(tmp_path, fake_denoisers, no_asr, monkeypatch):
    closed = []
    monkeypatch.setattr(NoiseReductionService, 'close', lambda self: closed.append(True))
    s = settings(tmp_path, asr_enabled=False, noise_enabled=True)
    asyncio.run(with_running_server(s, noise_ok))
    assert closed == [True]


def test_missing_noise_model_stops_the_start_before_any_model_loads(tmp_path, no_asr):
    s = settings(tmp_path, asr_enabled=False, noise_enabled=True,
                 noise_model=str(tmp_path / 'nope.onnx'))
    with pytest.raises(FileNotFoundError, match='nope.onnx'):
        asyncio.run(server.serve(s))


def test_dpdfnet2_factory_names_the_missing_package(tmp_path, monkeypatch):
    model = tmp_path / 'dpdfnet2.onnx'
    model.write_bytes(b'onnx')
    monkeypatch.setitem(sys.modules, 'sherpa_onnx', None)          # import fails
    with pytest.raises(ImportError, match='sherpa-onnx'):
        denoise.dpdfnet2_factory(str(model), 1)


class _BrokenNativeLibrary(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == 'sherpa_onnx':
            raise ImportError('DLL load failed while importing _sherpa_onnx')
        return None


def test_dpdfnet2_factory_keeps_the_reason_of_an_import_failure(tmp_path, monkeypatch):
    # review L4: a native library that fails to load was reported as a missing package
    model = tmp_path / 'dpdfnet2.onnx'
    model.write_bytes(b'onnx')
    monkeypatch.delitem(sys.modules, 'sherpa_onnx', raising=False)
    monkeypatch.setattr(sys, 'meta_path', [_BrokenNativeLibrary(), *sys.meta_path])
    with pytest.raises(ImportError, match='DLL load failed'):
        denoise.dpdfnet2_factory(str(model), 1)


def fake_sherpa(valid=True, rate=16000):
    """A stand-in sherpa_onnx module with the classes dpdfnet2_factory uses."""
    module = types.ModuleType('sherpa_onnx')

    class Config:
        def __init__(self, **values):
            self.values = values

    class OnlineConfig(Config):
        def validate(self):
            return valid

    class Denoiser:
        sample_rate = rate

        def __init__(self, config):
            self.config = config
    module.OfflineSpeechDenoiserDpdfNetModelConfig = Config
    module.OfflineSpeechDenoiserModelConfig = Config
    module.OnlineSpeechDenoiserConfig = OnlineConfig
    module.OnlineSpeechDenoiser = Denoiser
    return module


def test_dpdfnet2_factory_refuses_a_rejected_configuration_and_a_non_16khz_model(tmp_path, monkeypatch):
    # review L4: these two branches were never run without the real package
    model = tmp_path / 'dpdfnet2.onnx'
    model.write_bytes(b'onnx')
    monkeypatch.setitem(sys.modules, 'sherpa_onnx', fake_sherpa(valid=False))
    with pytest.raises(ValueError, match='rejected'):
        denoise.dpdfnet2_factory(str(model), 1)
    monkeypatch.setitem(sys.modules, 'sherpa_onnx', fake_sherpa(rate=8000))
    make = denoise.dpdfnet2_factory(str(model), 2)
    with pytest.raises(ValueError, match='8000 Hz'):
        make()
    monkeypatch.setitem(sys.modules, 'sherpa_onnx', fake_sherpa())
    made = denoise.dpdfnet2_factory(str(model), 2)()
    model_config = made.config.values['model'].values
    assert model_config['num_threads'] == 2 and model_config['provider'] == 'cpu'
    assert model_config['dpdfnet'].values['model'] == str(model)


def test_noise_startup_failure_is_logged(tmp_path, no_asr, monkeypatch, caplog):
    # review L4: only the traceback in logs/asr-server.out told why the server stopped
    monkeypatch.setitem(sys.modules, 'sherpa_onnx', None)
    s = settings(tmp_path, asr_enabled=False, noise_enabled=True)
    with caplog.at_level(logging.ERROR), pytest.raises(ImportError):
        asyncio.run(server.serve(s))
    assert any(r.levelno == logging.ERROR and 'noise_reduction cannot start' in r.getMessage()
               and 'sherpa-onnx' in r.getMessage() for r in caplog.records)


def test_missing_noise_model_stops_the_start_before_asr_loads(tmp_path, no_asr):
    # review L4: the order (denoisers before the GPU model) was only tested with ASR off
    s = settings(tmp_path, noise_enabled=True, noise_model=str(tmp_path / 'nope.onnx'))
    with pytest.raises(FileNotFoundError, match='nope.onnx'):
        asyncio.run(server.serve(s))


def test_noise_service_is_closed_when_asr_fails_to_load(tmp_path, fake_denoisers, monkeypatch):
    closed = []
    monkeypatch.setattr(NoiseReductionService, 'close', lambda self: closed.append(True))

    def asr_fails(args):
        raise RuntimeError('CUDA out of memory')
    monkeypatch.setattr(server, '_build_asr', asr_fails)
    with pytest.raises(RuntimeError, match='CUDA'):
        asyncio.run(server.serve(settings(tmp_path, noise_enabled=True)))
    assert closed == [True]


def test_noise_service_is_closed_even_when_closing_asr_fails(tmp_path, fake_denoisers, monkeypatch):
    # review L4: an error in the ASR close skipped the noise close
    closed = []
    monkeypatch.setattr(NoiseReductionService, 'close', lambda self: closed.append(True))

    class Core:
        def close(self):
            raise RuntimeError('executor broken')
    monkeypatch.setattr(server, '_build_asr', lambda args: Core())

    async def started(stub):
        pass
    with pytest.raises(RuntimeError, match='executor broken'):
        asyncio.run(with_running_server(settings(tmp_path, noise_enabled=True), started))
    assert closed == [True]
