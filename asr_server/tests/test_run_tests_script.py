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


def isolated_files():
    text = (ROOT / 'scripts' / 'run_tests.sh').read_text(encoding='utf-8')
    match = re.search(r'^GRPC_SERVER_TESTS="([^"]*)"', text, re.M)
    assert match, 'GRPC_SERVER_TESTS not found in scripts/run_tests.sh'
    return set(match.group(1).split())


def test_every_file_that_starts_a_grpc_server_runs_in_its_own_process():
    starting = {f'tests/{path.name}' for path in (ROOT / 'tests').glob('test_*.py')
                if path.name != Path(__file__).name
                and STARTS_A_SERVER.search(path.read_text(encoding='utf-8'))}
    assert starting == isolated_files()
    assert all((ROOT / name).is_file() for name in isolated_files())
