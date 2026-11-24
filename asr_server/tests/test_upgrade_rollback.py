"""scripts/update.sh --restart and bin/rollback.sh: every script is the real one, only the model
server is replaced by a small stand-in that listens on the configured port."""
import os
import shutil
import socket
import subprocess
import sys

import pytest

from test_deploy_scripts import script, source, update  # noqa: F401  (source is a fixture)

pytestmark = pytest.mark.skipif(os.name != 'posix' or shutil.which('bash') is None,
                                reason='the deployment scripts need bash')

SERVER = '''
import asyncio
from pathlib import Path


async def serve(settings):
    Path('logs/running_version').write_text({version!r})
    behaviour = {behaviour!r}
    if behaviour == 'crash':
        await asyncio.sleep(1.5)
        raise RuntimeError('model failed to load')
    if behaviour == 'hang':
        await asyncio.Event().wait()
    server = await asyncio.start_server(lambda r, w: w.close(), settings.host, settings.port)
    async with server:
        await server.serve_forever()
'''
BASE_ENV = {'PYTHON': sys.executable, 'ASR_START_WAIT_SEC': '20'}


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture
def home(source, tmp_path):  # noqa: F811
    root = tmp_path / 'app'
    (root / 'model' / 'Qwen3-ASR-1.7B').mkdir(parents=True)
    (root / 'model' / 'punctuation-restoration-small').mkdir()
    (root / 'model' / 'silero_vad.jit').write_bytes(b'x')
    config = source / 'config' / 'server.yaml'
    config.write_text(config.read_text().replace('port: 50051', f'port: {free_port()}'))
    yield root
    if (root / 'bin' / 'stop.sh').exists():
        subprocess.run(['bash', str(root / 'bin' / 'stop.sh'), '--force'], capture_output=True,
                       env={**os.environ, **BASE_ENV})


def install(source, home, version, behaviour='good', args=('--restart',), **env):  # noqa: F811
    (source / 'src' / 'asr_server' / 'server.py').write_text(
        SERVER.format(version=version, behaviour=behaviour))
    return update(source, home, *args, env={**BASE_ENV, **env})


def rollback(home, *args, **env):
    return subprocess.run(['bash', str(home / 'bin' / 'rollback.sh'), *args], capture_output=True,
                          text=True, env={**os.environ, **BASE_ENV, **env})


def saved(home):
    return sorted(p.name for p in (home / 'backup').iterdir() if p.name != 'rolled-back')


def installed_code(home):
    return (home / 'lib' / 'asr_server' / 'server.py').read_text()


def running_version(home):
    status = subprocess.run(['bash', str(home / 'bin' / 'status.sh')], capture_output=True,
                            env={**os.environ, **BASE_ENV})
    if status.returncode != 0:
        return None
    return (home / 'logs' / 'running_version').read_text()


def health(path):
    lines = (path / 'VERSION').read_text().splitlines()
    return [line for line in lines if line.startswith('health=')]


def add_key_to_the_new_version(source):  # noqa: F811
    """The next release accepts asr.only_new; the installed code rejects it as an unknown key."""
    path = source / 'src' / 'asr_server' / 'config.py'
    old = path.read_text()
    new = old.replace("    asr_log_results: bool = False ",
                      "    asr_only_new: bool = False\n    asr_log_results: bool = False ")
    new = new.replace("('asr', 'log_results'): 'asr_log_results',",
                      "('asr', 'log_results'): 'asr_log_results', ('asr', 'only_new'): 'asr_only_new',")
    assert new.count('asr_only_new') == 2
    path.write_text(new)


def set_new_key(home):
    config = home / 'config' / 'server.yaml'
    config.write_text(config.read_text().replace('asr:\n', 'asr:\n  only_new: true\n', 1))


# --- what an update saves ---

def test_the_build_ships_the_upgrade_scripts(source, home):  # noqa: F811
    assert script(source, 'build.sh').returncode == 0
    assert install(source, home, 'v1').returncode == 0
    for name in ('rollback.sh', 'wait_ready.sh'):
        assert os.access(source / 'build' / 'app' / 'bin' / name, os.X_OK)
        assert os.access(home / 'bin' / name, os.X_OK)


def test_the_first_install_saves_nothing_and_restart_starts_the_server(source, home):  # noqa: F811
    result = install(source, home, 'v1')
    assert result.returncode == 0, result.stderr
    assert saved(home) == []
    assert running_version(home) == 'v1'
    assert health(home) == ['health=ok']


def test_an_update_saves_the_version_it_replaces(source, home):  # noqa: F811
    assert install(source, home, 'v1').returncode == 0
    config = home / 'config' / 'server.yaml'
    config.write_text(config.read_text() + '\n# local setting of v1\n')
    (home / 'bin' / 'env.sh').write_text('# env of v1\n')
    dist_of_v1 = (home / 'config' / 'server.yaml.dist').read_text()
    shipped = source / 'config' / 'server.yaml'
    shipped.write_text(shipped.read_text() + '\n# default of v2\n')

    result = install(source, home, 'v2')
    assert result.returncode == 0, result.stderr
    assert running_version(home) == 'v2'
    [backup] = saved(home)
    copy = home / 'backup' / backup
    assert "'v1'" in (copy / 'lib' / 'asr_server' / 'server.py').read_text()
    assert '# local setting of v1' in (copy / 'config' / 'server.yaml').read_text()
    assert (copy / 'config' / 'server.yaml.dist').read_text() == dist_of_v1
    assert (copy / 'bin' / 'env.sh').read_text() == '# env of v1\n'
    assert health(copy) == ['health=ok']


def test_at_most_ten_versions_are_kept(source, home):  # noqa: F811
    assert install(source, home, 'v1').returncode == 0
    for day in range(1, 13):
        (home / 'backup' / f'201001{day:02d}-000000').mkdir()
    (home / 'backup' / 'notes').mkdir()

    assert install(source, home, 'v2').returncode == 0
    names = saved(home)
    assert 'notes' in names                     # only saved versions are removed
    names.remove('notes')
    assert len(names) == 10
    assert names[0] == '20100104-000000'


def test_the_number_of_versions_is_read_from_env_sh(source, home):  # noqa: F811
    assert install(source, home, 'v1').returncode == 0
    (home / 'bin' / 'env.sh').write_text('export ASR_BACKUP_KEEP=1\n')
    assert install(source, home, 'v2').returncode == 0
    first = saved(home)
    assert install(source, home, 'v3').returncode == 0
    assert len(saved(home)) == 1 and saved(home) != first


@pytest.mark.parametrize('name, value', [
    ('ASR_BACKUP_KEEP', '0'),
    ('ASR_BACKUP_KEEP', 'ten'),
    ('ASR_BACKUP_KEEP', '010'),                 # bash would read it as octal 8
    ('ASR_START_WAIT_SEC', 'soon'),
    ('ASR_START_WAIT_SEC', '08'),               # bash arithmetic rejects it
])
def test_a_wrong_setting_changes_nothing(source, home, name, value):  # noqa: F811
    assert install(source, home, 'v1').returncode == 0
    result = install(source, home, 'v2', **{name: value})
    assert result.returncode == 2
    assert name in result.stderr
    assert "'v1'" in installed_code(home) and running_version(home) == 'v1'
    assert saved(home) == []


def test_a_configuration_the_new_code_rejects_saves_nothing(source, home):  # noqa: F811
    assert install(source, home, 'v1').returncode == 0
    config = home / 'config' / 'server.yaml'
    config.write_text(config.read_text() + '\nremoved_section:\n  key: 1\n')
    result = install(source, home, 'v2')
    assert result.returncode == 1
    assert list((home / 'backup').iterdir()) == []
    assert running_version(home) == 'v1'


def test_an_update_stops_when_the_saved_version_could_not_be_restored(source, home):  # noqa: F811
    assert install(source, home, 'v1').returncode == 0
    add_key_to_the_new_version(source)
    set_new_key(home)

    result = install(source, home, 'v2')
    assert result.returncode == 1
    assert 'unknown key asr.only_new' in result.stderr
    assert 'could not be rolled back' in result.stderr
    assert "'v1'" in installed_code(home) and running_version(home) == 'v1'
    assert list((home / 'backup').iterdir()) == []

    result = install(source, home, 'v2', args=('--restart', '--ignore-old-config'))
    assert result.returncode == 0, result.stderr
    assert running_version(home) == 'v2' and len(saved(home)) == 1


def test_status_reports_a_running_server_whose_configuration_became_invalid(source, home):  # noqa: F811
    assert install(source, home, 'v1').returncode == 0
    config = home / 'config' / 'server.yaml'
    config.write_text(config.read_text() + '\nremoved_section:\n  key: 1\n')
    status = subprocess.run(['bash', str(home / 'bin' / 'status.sh')], capture_output=True, text=True,
                            env={**os.environ, **BASE_ENV})
    assert status.returncode == 0, status.stderr
    assert 'is running' in status.stdout


def test_an_update_stops_a_server_that_an_older_status_script_misreports(source, home):  # noqa: F811
    assert install(source, home, 'v1').returncode == 0
    # bin/status.sh of the versions already installed exits with 2 on a configuration it rejects.
    (home / 'bin' / 'status.sh').write_text('exit 2\n')
    add_key_to_the_new_version(source)
    set_new_key(home)

    result = install(source, home, 'v2', args=('--restart', '--ignore-old-config'))
    assert result.returncode == 0, result.stderr
    assert running_version(home) == 'v2'


# --- the check after the restart: an error is reported, never rolled back ---

def test_a_server_that_dies_while_loading_is_reported_and_not_rolled_back(source, home):  # noqa: F811
    assert install(source, home, 'v1').returncode == 0
    result = install(source, home, 'v2', behaviour='crash')
    assert result.returncode == 1
    assert 'NOT rolled back' in result.stderr
    assert f'{home}/bin/rollback.sh' in result.stderr
    assert f'{home}/logs/asr-server.out' in result.stderr
    assert "'v2'" in installed_code(home)
    assert running_version(home) is None
    assert health(home) == ['health=failed']


def test_a_server_that_never_opens_its_port_is_left_running(source, home):  # noqa: F811
    assert install(source, home, 'v1').returncode == 0
    result = install(source, home, 'v2', behaviour='hang', ASR_START_WAIT_SEC='3')
    assert result.returncode == 1
    assert 'NOT rolled back' in result.stderr
    assert running_version(home) == 'v2'        # kept alive so that it can be examined
    assert health(home) == ['health=failed']


def test_the_check_can_be_turned_off(source, home):  # noqa: F811
    assert install(source, home, 'v1').returncode == 0
    result = install(source, home, 'v2', behaviour='hang', ASR_START_WAIT_SEC='0')
    assert result.returncode == 0, result.stderr
    assert 'readiness not checked' in result.stdout
    assert health(home) == []


def test_restart_starts_the_fixed_version_after_a_failed_update(source, home):  # noqa: F811
    assert install(source, home, 'v1').returncode == 0
    assert install(source, home, 'v2', behaviour='crash').returncode == 1
    assert running_version(home) is None

    result = install(source, home, 'v3')
    assert result.returncode == 0, result.stderr
    assert running_version(home) == 'v3'
    assert health(home) == ['health=ok']


def test_restart_of_a_running_server_without_its_models_changes_nothing(source, home):  # noqa: F811
    # review: the running server was stopped, replaced and started again without its models
    assert install(source, home, 'v1').returncode == 0
    assert running_version(home) == 'v1'
    shutil.rmtree(home / 'model' / 'Qwen3-ASR-1.7B')
    result = install(source, home, 'v2')
    assert result.returncode == 1
    assert 'MISSING  ASR model directory' in result.stdout
    assert 'the application was not updated' in result.stderr
    assert running_version(home) == 'v1' and "'v1'" in installed_code(home)
    assert saved(home) == []


def test_restart_does_not_start_without_the_models(source, home):  # noqa: F811
    shutil.rmtree(home / 'model')
    result = install(source, home, 'v1')
    assert result.returncode == 0, result.stderr
    assert 'copy the missing models' in result.stdout
    assert running_version(home) is None


def test_wait_ready_reports_a_stopped_server(source, home):  # noqa: F811
    assert install(source, home, 'v1', args=()).returncode == 0
    result = subprocess.run(['bash', str(home / 'bin' / 'wait_ready.sh')], capture_output=True,
                            text=True, env={**os.environ, **BASE_ENV})
    assert result.returncode == 3
    assert 'not running' in result.stderr


# --- rollback ---

def test_rollback_restores_the_previous_version(source, home):  # noqa: F811
    assert install(source, home, 'v1').returncode == 0
    config = home / 'config' / 'server.yaml'
    config_of_v1 = config.read_text()
    env_of_v1 = (home / 'bin' / 'env.sh').read_text()
    version_of_v1 = (home / 'VERSION').read_text()
    assert install(source, home, 'v2').returncode == 0
    config.write_text(config_of_v1 + '\n# setting of v2\n')
    (home / 'bin' / 'env.sh').write_text(env_of_v1 + '\n# env of v2\n')

    result = rollback(home)
    assert result.returncode == 0, result.stderr
    assert running_version(home) == 'v1'
    assert "'v1'" in installed_code(home)
    assert config.read_text() == config_of_v1
    assert (home / 'bin' / 'env.sh').read_text() == env_of_v1
    assert (home / 'VERSION').read_text() == version_of_v1
    # What was replaced can still be examined.
    [replaced] = list((home / 'backup' / 'rolled-back').iterdir())
    assert "'v2'" in (replaced / 'lib' / 'asr_server' / 'server.py').read_text()
    assert '# setting of v2' in (replaced / 'config' / 'server.yaml').read_text()
    assert '# env of v2' in (replaced / 'bin' / 'env.sh').read_text()


def test_rollback_works_when_the_configuration_has_a_key_of_the_new_version(source, home):  # noqa: F811
    assert install(source, home, 'v1').returncode == 0
    add_key_to_the_new_version(source)
    assert install(source, home, 'v2').returncode == 0
    set_new_key(home)

    result = rollback(home)
    assert result.returncode == 0, result.stderr
    assert running_version(home) == 'v1'
    assert 'only_new' not in (home / 'config' / 'server.yaml').read_text()
    [replaced] = list((home / 'backup' / 'rolled-back').iterdir())
    assert 'only_new' in (replaced / 'config' / 'server.yaml').read_text()


def test_rollback_can_be_run_twice(source, home):  # noqa: F811
    assert install(source, home, 'v1').returncode == 0
    assert install(source, home, 'v2').returncode == 0
    assert rollback(home).returncode == 0
    result = rollback(home)
    assert result.returncode == 0, result.stderr
    assert running_version(home) == 'v1'


def test_rollback_skips_a_version_that_failed_its_check(source, home):  # noqa: F811
    assert install(source, home, 'v1').returncode == 0
    assert install(source, home, 'v2', behaviour='crash').returncode == 1
    assert install(source, home, 'v3', behaviour='crash').returncode == 1
    assert len(saved(home)) == 2

    result = rollback(home)
    assert result.returncode == 0, result.stderr
    assert saved(home)[0] in result.stdout
    assert running_version(home) == 'v1'
    assert health(home) == ['health=ok']


def test_rollback_to_a_named_version(source, home):  # noqa: F811
    assert install(source, home, 'v1').returncode == 0
    assert install(source, home, 'v2').returncode == 0
    assert install(source, home, 'v3').returncode == 0
    oldest, newest = saved(home)
    assert rollback(home, oldest).returncode == 0
    assert running_version(home) == 'v1'
    assert rollback(home, newest).returncode == 0
    assert running_version(home) == 'v2'


def test_list_shows_the_versions_and_changes_nothing(source, home):  # noqa: F811
    assert install(source, home, 'v1').returncode == 0
    assert install(source, home, 'v2', behaviour='crash').returncode == 1
    assert install(source, home, 'v3').returncode == 0
    good, failed = saved(home)

    result = rollback(home, '--list')
    assert result.returncode == 0, result.stderr
    lines = {line.split()[0]: line for line in result.stdout.splitlines() if line.strip()}
    assert 'ok' in lines[good].split() and '<-' in lines[good]
    assert 'failed' in lines[failed].split() and '<-' not in lines[failed]
    assert running_version(home) == 'v3'


@pytest.mark.parametrize('name', ['20000101-000000', '../../lib', '-x'])
def test_rollback_refuses_an_unknown_version(source, home, name):  # noqa: F811
    assert install(source, home, 'v1').returncode == 0
    assert install(source, home, 'v2').returncode == 0
    result = rollback(home, name)
    assert result.returncode == 2
    assert running_version(home) == 'v2' and "'v2'" in installed_code(home)


def test_rollback_without_a_saved_version(source, home):  # noqa: F811
    assert install(source, home, 'v1').returncode == 0
    result = rollback(home)
    assert result.returncode == 1
    assert 'no saved version' in result.stderr
    assert running_version(home) == 'v1'


def test_rollback_changes_nothing_when_the_saved_configuration_is_invalid(source, home):  # noqa: F811
    assert install(source, home, 'v1').returncode == 0
    assert install(source, home, 'v2').returncode == 0
    [backup] = saved(home)
    (home / 'backup' / backup / 'config' / 'server.yaml').write_text('removed_section: 1\n')

    result = rollback(home)
    assert result.returncode == 1
    assert 'nothing was changed' in result.stderr
    assert running_version(home) == 'v2' and "'v2'" in installed_code(home)


# --- found by the independent review ---

def test_repeated_failed_updates_keep_the_last_working_version(source, home):  # noqa: F811
    assert install(source, home, 'v1').returncode == 0
    for version in ('v2', 'v3', 'v4'):
        assert install(source, home, version, behaviour='crash', ASR_BACKUP_KEEP='2').returncode == 1
    assert len(saved(home)) == 2

    result = rollback(home)
    assert result.returncode == 0, result.stderr
    assert running_version(home) == 'v1'


def test_an_incomplete_build_changes_nothing(source, home):  # noqa: F811
    assert install(source, home, 'v1').returncode == 0
    assert script(source, 'build.sh').returncode == 0
    (source / 'build' / 'app' / 'bin' / 'rollback.sh').unlink()

    result = update(source, home, '--restart', '--no-build', env=BASE_ENV)
    assert result.returncode == 1
    assert 'run scripts/build.sh' in result.stderr
    assert running_version(home) == 'v1' and saved(home) == []


def test_a_second_rollback_keeps_what_the_first_one_replaced(source, home):  # noqa: F811
    assert install(source, home, 'v1').returncode == 0
    assert install(source, home, 'v2').returncode == 0
    config = home / 'config' / 'server.yaml'
    config.write_text(config.read_text() + '\n# setting of v2\n')

    assert rollback(home).returncode == 0
    assert rollback(home).returncode == 0
    copies = list((home / 'backup' / 'rolled-back').iterdir())
    assert any('# setting of v2' in (c / 'config' / 'server.yaml').read_text()
               and "'v2'" in (c / 'lib' / 'asr_server' / 'server.py').read_text() for c in copies)


def test_replaced_versions_are_limited_too(source, home):  # noqa: F811
    assert install(source, home, 'v1').returncode == 0
    assert install(source, home, 'v2').returncode == 0
    assert rollback(home, ASR_BACKUP_KEEP='1').returncode == 0
    assert rollback(home, ASR_BACKUP_KEEP='1').returncode == 0
    assert len(list((home / 'backup' / 'rolled-back').iterdir())) == 1


def test_a_listener_of_another_process_does_not_count_as_ready(source, home):  # noqa: F811
    assert install(source, home, 'v1', args=()).returncode == 0
    text = (home / 'config' / 'server.yaml').read_text()
    port = int(text.split('port: ', 1)[1].split()[0])
    with socket.socket() as other:
        other.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        other.bind(('127.0.0.1', port))
        other.listen()
        result = install(source, home, 'v2', behaviour='hang', ASR_START_WAIT_SEC='3')
    assert result.returncode == 1
    assert health(home) == ['health=failed']


def test_an_update_repairs_a_damaged_install(source, home):  # noqa: F811
    assert install(source, home, 'v1').returncode == 0
    subprocess.run(['bash', str(home / 'bin' / 'stop.sh')], check=True, capture_output=True,
                   env={**os.environ, **BASE_ENV})
    (home / 'bin' / '_common.sh').unlink()

    result = install(source, home, 'v2')
    assert result.returncode == 0, result.stderr
    assert running_version(home) == 'v2'


def test_rollback_prefers_a_version_that_passed_its_check(source, home):  # noqa: F811
    assert install(source, home, 'v1').returncode == 0
    subprocess.run(['bash', str(home / 'bin' / 'stop.sh')], check=True, capture_output=True,
                   env={**os.environ, **BASE_ENV})
    assert install(source, home, 'v2', behaviour='crash', args=()).returncode == 0   # never started
    assert install(source, home, 'v3').returncode == 0

    result = rollback(home)
    assert result.returncode == 0, result.stderr
    assert running_version(home) == 'v1'


def test_rollback_skips_a_saved_version_that_cannot_read_its_configuration(source, home):  # noqa: F811
    assert install(source, home, 'v1').returncode == 0
    assert install(source, home, 'v2').returncode == 0
    add_key_to_the_new_version(source)
    set_new_key(home)
    result = install(source, home, 'v3', args=('--restart', '--ignore-old-config'))
    assert result.returncode == 0, result.stderr

    result = rollback(home)
    assert result.returncode == 0, result.stderr
    assert running_version(home) == 'v1'


def test_a_refused_update_keeps_the_defaults_of_the_installed_version(source, home):  # noqa: F811
    assert install(source, home, 'v1').returncode == 0
    dist = home / 'config' / 'server.yaml.dist'
    dist_of_v1 = dist.read_text()
    shipped = source / 'config' / 'server.yaml'
    shipped.write_text(shipped.read_text() + '\n# default of v2\n')
    config = home / 'config' / 'server.yaml'
    good_config = config.read_text()
    config.write_text(good_config + '\nremoved_section:\n  key: 1\n')

    result = install(source, home, 'v2')
    assert result.returncode == 1
    assert dist.read_text() == dist_of_v1
    assert '# default of v2' in (home / 'config' / 'server.yaml.dist.new').read_text()
    assert 'server.yaml.dist.new' in result.stderr

    config.write_text(good_config)
    assert install(source, home, 'v2').returncode == 0
    [backup] = saved(home)
    assert (home / 'backup' / backup / 'config' / 'server.yaml.dist').read_text() == dist_of_v1
    assert '# default of v2' in dist.read_text()
    assert not (home / 'config' / 'server.yaml.dist.new').exists()


def test_help_shows_the_whole_usage(source):  # noqa: F811
    result = script(source, 'update.sh', '--help')
    assert result.returncode == 0
    assert 'NOT rolled back' in result.stdout
    assert 'set -euo' not in result.stdout


def test_rollback_reports_a_restored_version_that_does_not_start(source, home):  # noqa: F811
    assert install(source, home, 'v1', behaviour='crash', ASR_START_WAIT_SEC='0').returncode == 0
    assert install(source, home, 'v2').returncode == 0
    result = rollback(home)
    assert result.returncode == 1
    assert 'did not become ready' in result.stderr
    assert health(home) == ['health=failed']
