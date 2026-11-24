"""scripts/build.sh and scripts/update.sh: layout of the run directory, what an update
replaces and what it keeps, and the cases in which nothing may be changed."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(os.name != 'posix' or shutil.which('bash') is None,
                                reason='the deployment scripts need bash')


@pytest.fixture
def source(tmp_path):
    """A private copy of the source tree, so that build/ is not written into the checkout."""
    dst = tmp_path / 'source'
    for name in ('bin', 'config', 'scripts', 'src'):
        shutil.copytree(ROOT / name, dst / name,
                        ignore=shutil.ignore_patterns('__pycache__', 'env.sh'))
    shutil.copy(ROOT / 'pyproject.toml', dst)
    return dst


def script(source, name, *args, env=None):
    base = {k: v for k, v in os.environ.items() if k != 'ASR_APP_HOME'}
    full_env = {**base, 'PYTHON': sys.executable, **(env or {})}
    return subprocess.run(['bash', str(source / 'scripts' / name), *args],
                          capture_output=True, text=True, env=full_env)


def update(source, home, *args, env=None):
    return script(source, 'update.sh', '--app-home', str(home), *args, env=env)


def test_build_makes_the_run_directory_layout(source):
    result = script(source, 'build.sh')
    assert result.returncode == 0, result.stderr
    stage = source / 'build' / 'app'
    assert (stage / 'lib' / 'asr_server' / 'server.py').is_file()
    assert (stage / 'lib' / 'asr_server' / 'proto' / 'speech.proto').is_file()
    assert not list(stage.rglob('__pycache__'))
    for name in ('start.sh', 'stop.sh', 'status.sh', '_common.sh'):
        assert os.access(stage / 'bin' / name, os.X_OK)
    assert (stage / 'config' / 'server.yaml').is_file()
    version = (stage / 'VERSION').read_text()
    assert 'name=asr_server' in version and 'version=0.1.0' in version


def test_build_stops_on_a_syntax_error(source):
    (source / 'src' / 'asr_server' / 'broken.py').write_text('def broken(:\n')
    result = script(source, 'build.sh')
    assert result.returncode != 0
    assert 'SyntaxError' in result.stderr


def test_first_install(source, tmp_path):
    home = tmp_path / 'app'
    result = update(source, home)
    assert result.returncode == 0, result.stderr
    for name in ('bin', 'config', 'lib', 'logs', 'model', 'data'):
        assert (home / name).is_dir(), name
    assert (home / 'lib' / 'asr_server' / '__main__.py').is_file()
    assert os.access(home / 'bin' / 'start.sh', os.X_OK)
    assert (home / 'bin' / 'env.sh').is_file()
    assert 'installed_at=' in (home / 'VERSION').read_text()

    config = (home / 'config' / 'server.yaml').read_text()
    assert f'model: {home}/model/Qwen3-ASR-1.7B' in config
    assert f'silero_model: {home}/model/silero_vad.jit' in config
    assert f'model: {home}/model/punctuation-restoration-small' in config
    assert f'model: {home}/model/dpdfnet2.onnx' in config
    assert '/opt/models' not in config
    assert (home / 'config' / 'logging.yaml').is_file()

    # The models are not there yet: reported, and the install still succeeds.
    assert f'MISSING  ASR model directory: {home}/model/Qwen3-ASR-1.7B' in result.stdout
    assert f'MISSING  punctuation model directory: {home}/model/punctuation-restoration-small' in result.stdout
    assert f'MISSING  Silero VAD file: {home}/model/silero_vad.jit' in result.stdout
    # noise_reduction is off by default: its model is not asked for
    assert 'DPDFNet-2' not in result.stdout and 'sherpa-onnx' not in result.stdout


def _set_tasks(home, asr, noise):
    """Switch the tasks in the run directory's server.yaml (as shipped, after a first install)."""
    config = home / 'config' / 'server.yaml'
    text = config.read_text()
    asr_line = '  enabled: true              # task speech_to_text'
    noise_line = '  enabled: false             # false: noise_reduction ends with UNIMPLEMENTED'
    assert asr_line in text and noise_line in text
    text = text.replace(asr_line, asr_line.replace('true ', 'false', 1) if not asr else asr_line)
    text = text.replace(noise_line, noise_line.replace('false', 'true ', 1) if noise else noise_line)
    config.write_text(text)


def _sherpa_installed():
    return subprocess.run([sys.executable, '-c', 'import sherpa_onnx'], capture_output=True).returncode == 0


def test_noise_only_install_asks_for_no_asr_model(source, tmp_path):
    home = tmp_path / 'app'
    assert update(source, home).returncode == 0
    _set_tasks(home, asr=False, noise=True)
    (home / 'model' / 'dpdfnet2.onnx').write_bytes(b'onnx')
    result = update(source, home, '--no-build')
    assert result.returncode == 0, result.stderr
    assert 'ASR model' not in result.stdout and 'punctuation model' not in result.stdout
    assert 'Silero' not in result.stdout and 'DPDFNet-2' not in result.stdout
    if _sherpa_installed():
        assert 'MISSING' not in result.stdout and 'copy the missing models' not in result.stdout
        assert f'start with: {home}/bin/start.sh' in result.stdout
    else:
        assert 'MISSING  Python package sherpa-onnx' in result.stdout


def test_missing_noise_model_is_reported(source, tmp_path):
    home = tmp_path / 'app'
    assert update(source, home).returncode == 0
    _set_tasks(home, asr=True, noise=True)
    result = update(source, home, '--no-build')
    assert result.returncode == 0, result.stderr
    assert f'MISSING  DPDFNet-2 model file: {home}/model/dpdfnet2.onnx' in result.stdout
    assert f'MISSING  ASR model directory: {home}/model/Qwen3-ASR-1.7B' in result.stdout
    assert 'copy the missing models' in result.stdout


def _set_asr_denoise(home, model=None):
    """Turn asr_denoise on in the run directory's server.yaml; `model` replaces its model path."""
    config = home / 'config' / 'server.yaml'
    text = config.read_text()
    line = '  enabled: false             # false: VAD and ASR get the audio as sent.'
    assert line in text
    text = text.replace(line, line.replace('false', 'true ', 1))
    if model is not None:
        section = text.index('asr_denoise:')
        text = text[:section] + text[section:].replace(f'model: {home}/model/dpdfnet2.onnx', f'model: {model}', 1)
    config.write_text(text)


def test_missing_asr_denoise_model_is_reported(source, tmp_path):
    home = tmp_path / 'app'
    assert update(source, home).returncode == 0
    _set_asr_denoise(home, model=f'{home}/model/stt-dpdfnet2.onnx')
    result = update(source, home, '--no-build')
    assert result.returncode == 0, result.stderr
    assert f'MISSING  DPDFNet-2 model file (asr_denoise.model): {home}/model/stt-dpdfnet2.onnx' in result.stdout
    if not _sherpa_installed():
        assert 'MISSING  Python package sherpa-onnx==2.0.2 (asr_denoise.enabled;' in result.stdout
    assert 'copy the missing models' in result.stdout


def test_one_sherpa_report_for_both_denoising_settings(source, tmp_path):
    home = tmp_path / 'app'
    assert update(source, home).returncode == 0
    _set_tasks(home, asr=True, noise=True)
    _set_asr_denoise(home)
    (home / 'model' / 'dpdfnet2.onnx').write_bytes(b'onnx')
    result = update(source, home, '--no-build')
    assert result.returncode == 0, result.stderr
    assert 'DPDFNet-2 model file' not in result.stdout          # one file, present, for both
    if not _sherpa_installed():
        assert result.stdout.count('MISSING  Python package sherpa-onnx') == 1
        assert '(noise_reduction.enabled and asr_denoise.enabled;' in result.stdout


def _set_model_paths(home, asr, punct):
    config = home / 'config' / 'server.yaml'
    text = config.read_text()
    text = text.replace(f'model: {home}/model/Qwen3-ASR-1.7B', f'model: {asr}')
    text = text.replace(f'model: {home}/model/punctuation-restoration-small', f'model: {punct}')
    config.write_text(text)


def test_relative_model_paths_are_checked_from_the_run_directory(source, tmp_path):
    # review: they were checked from the directory update.sh was started in; the server runs in
    # the run directory (bin/start.sh)
    home = tmp_path / 'app'
    assert update(source, home).returncode == 0
    (home / 'model' / 'Qwen3-ASR-1.7B').mkdir()
    (home / 'model' / 'silero_vad.jit').write_bytes(b'x')
    _set_model_paths(home, 'model/Qwen3-ASR-1.7B', 'model/punct-missing')
    result = update(source, home, '--no-build')
    assert result.returncode == 0, result.stderr
    assert 'ASR model' not in result.stdout
    assert f'MISSING  punctuation model directory: {home}/model/punct-missing' in result.stdout


def test_a_model_path_with_a_quote_is_checked(source, tmp_path):
    # review: --check-config prints such a path in double quotes, which the parser skipped
    home = tmp_path / 'app'
    assert update(source, home).returncode == 0
    (home / 'model' / 'Qwen3-ASR-1.7B').mkdir()
    (home / 'model' / 'silero_vad.jit').write_bytes(b'x')
    _set_model_paths(home, f'{home}/model/Qwen3-ASR-1.7B', f'"{home}/model/o\'brien"')
    result = update(source, home, '--no-build')
    assert result.returncode == 0, result.stderr
    assert f"MISSING  punctuation model directory: {home}/model/o'brien" in result.stdout


def test_no_missing_report_when_the_models_are_present(source, tmp_path):
    home = tmp_path / 'app'
    (home / 'model' / 'Qwen3-ASR-1.7B').mkdir(parents=True)
    (home / 'model' / 'punctuation-restoration-small').mkdir()
    (home / 'model' / 'silero_vad.jit').write_bytes(b'x')
    result = update(source, home)
    assert result.returncode == 0, result.stderr
    assert 'MISSING' not in result.stdout
    assert f'start with: {home}/bin/start.sh' in result.stdout


def test_update_replaces_code_and_keeps_local_files(source, tmp_path):
    home = tmp_path / 'app'
    assert update(source, home).returncode == 0

    config = home / 'config' / 'server.yaml'
    config.write_text(config.read_text().replace('port: 50051', 'port: 50999'))
    (home / 'bin' / 'env.sh').write_text('export CUDA_VISIBLE_DEVICES=1\n')
    (home / 'logs' / 'asr-server.log').write_text('old log\n')
    (home / 'model' / 'marker').write_text('model\n')
    (home / 'data' / 'marker').write_text('data\n')
    (home / 'lib' / 'asr_server' / 'only_in_old_version.py').write_text('')
    (source / 'src' / 'asr_server' / 'only_in_new_version.py').write_text('NEW = 1\n')

    result = update(source, home)
    assert result.returncode == 0, result.stderr
    assert (home / 'lib' / 'asr_server' / 'only_in_new_version.py').is_file()
    assert not (home / 'lib' / 'asr_server' / 'only_in_old_version.py').exists()
    assert (home / 'lib' / 'asr_server.prev' / 'only_in_old_version.py').is_file()
    assert not (home / 'lib' / 'asr_server.new').exists()

    assert 'port: 50999' in config.read_text()
    assert 'port: 50051' in (home / 'config' / 'server.yaml.dist').read_text()
    assert (home / 'bin' / 'env.sh').read_text() == 'export CUDA_VISIBLE_DEVICES=1\n'
    assert (home / 'logs' / 'asr-server.log').read_text() == 'old log\n'
    assert (home / 'model' / 'marker').is_file() and (home / 'data' / 'marker').is_file()
    assert f'kept     {config}' in result.stdout


def test_invalid_configuration_stops_the_update(source, tmp_path):
    home = tmp_path / 'app'
    assert update(source, home).returncode == 0
    config = home / 'config' / 'server.yaml'
    config.write_text(config.read_text() + '\nremoved_section:\n  key: 1\n')
    (source / 'src' / 'asr_server' / 'only_in_new_version.py').write_text('NEW = 1\n')

    result = update(source, home)
    assert result.returncode == 1
    assert 'the application was not updated' in result.stderr
    assert not (home / 'lib' / 'asr_server' / 'only_in_new_version.py').exists()
    assert not (home / 'lib' / 'asr_server.prev').exists()


def test_update_refuses_while_the_server_runs(source, tmp_path):
    home = tmp_path / 'app'
    assert update(source, home).returncode == 0
    (source / 'src' / 'asr_server' / 'only_in_new_version.py').write_text('NEW = 1\n')
    # A live process whose command line names asr_server, as bin/status.sh looks for.
    running = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)', 'asr_server'])
    try:
        (home / 'logs' / 'asr-server.pid').write_text(f'{running.pid}\n')
        result = update(source, home)
        assert result.returncode == 1
        assert 'use --restart' in result.stderr
        assert not (home / 'lib' / 'asr_server' / 'only_in_new_version.py').exists()
        assert running.poll() is None          # the running server was not touched
    finally:
        running.kill()
        running.wait()


def test_scripts_of_the_run_directory_use_lib(source, tmp_path):
    home = tmp_path / 'app'
    assert update(source, home).returncode == 0

    def code_dir(root):
        out = subprocess.run(['bash', '-c', f'source "{root}/bin/_common.sh"; echo "$CODE_DIR|$PYTHON"'],
                             capture_output=True, text=True, env={**os.environ, 'PYTHON': 'python-from-env'})
        assert out.returncode == 0, out.stderr
        return out.stdout.strip()

    assert code_dir(home) == f'{home}/lib|python-from-env'
    assert code_dir(source) == f'{source}/src|python-from-env'
    (home / 'bin' / 'env.sh').write_text('PYTHON=/opt/venv/bin/python\n')
    assert code_dir(home) == f'{home}/lib|/opt/venv/bin/python'

    (home / 'bin' / 'env.sh').write_text('')
    status = subprocess.run(['bash', str(home / 'bin' / 'status.sh')], capture_output=True, text=True,
                            env={**os.environ, 'PYTHON': sys.executable})
    assert status.returncode == 3 and 'not running' in status.stdout


def test_app_home_argument(source, tmp_path):
    assert update(source, 'relative/dir').returncode == 2
    assert script(source, 'update.sh', '--app-home', '/').returncode == 2
    assert script(source, 'update.sh', '--bogus').returncode == 2
    # Default taken from the environment.
    home = tmp_path / 'from-env'
    result = script(source, 'update.sh', env={'ASR_APP_HOME': str(home)})
    assert result.returncode == 0, result.stderr
    assert (home / 'lib' / 'asr_server').is_dir()


def test_no_build_needs_an_existing_build(source, tmp_path):
    result = update(source, tmp_path / 'app', '--no-build')
    assert result.returncode == 1
    assert 'run scripts/build.sh' in result.stderr
