"""scripts/run_tests.sh runs every test file that starts a real gRPC server in its own process.

On Windows a full run in one process sometimes hung in test_serve_noise.py when it ran after
test_regression.py::test_b2_cancel_during_inference_never_overlaps_model_calls: the gRPC aio
poller thread, shared by the whole process, was gone, so no later gRPC call got any event.
Never seen with one file per process. This test keeps the list in run_tests.sh complete.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# A server, a channel (same shared poller), or serve(), which builds a server unless a stub
# stops it first.
STARTS_A_SERVER = re.compile(r'grpc\.aio\.server\(|insecure_channel\(|\bserver\.serve\(|'
                             r'\bwith_server\(|\bwith_noise_server\(|\bwith_running_server\(')


def isolated_files(variable):
    text = (ROOT / 'scripts' / 'run_tests.sh').read_text(encoding='utf-8')
    match = re.search(rf'^{variable}="([^"]*)"', text, re.M)
    assert match, f'{variable} not found in scripts/run_tests.sh'
    return set(match.group(1).split())


def starting_a_server(directory):
    return {f'tests/{path.name}' for path in (directory / 'tests').glob('test_*.py')
            if path.name != Path(__file__).name
            and STARTS_A_SERVER.search(path.read_text(encoding='utf-8'))}


def test_every_file_that_starts_a_grpc_server_runs_in_its_own_process():
    assert starting_a_server(ROOT) == isolated_files('GRPC_SERVER_TESTS')
    assert all((ROOT / name).is_file() for name in isolated_files('GRPC_SERVER_TESTS'))


def test_the_eval_client_tests_that_start_a_grpc_server_run_in_their_own_process():
    # review: tools/eval/tests start real gRPC servers too, all in one process
    eval_root = ROOT / 'tools' / 'eval'
    assert starting_a_server(eval_root) == isolated_files('EVAL_GRPC_SERVER_TESTS')
    assert all((eval_root / name).is_file() for name in isolated_files('EVAL_GRPC_SERVER_TESTS'))
