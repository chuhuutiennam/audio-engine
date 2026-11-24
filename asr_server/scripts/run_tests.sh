#!/usr/bin/env bash
# Run the server tests, then the eval-client tests. Neither needs a GPU or model checkpoints.
# The two suites run in separate interpreters: each ships its own copy of speech_pb2.
# Extra arguments are pytest options (for example -k, -x); every run gets them. All runs go on
# after a failure; the exit status is the first failure's.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
cd "$ROOT"

# Test files of tests/ that start a real gRPC server (or call serve()) run each in their own
# interpreter. In one process, a full run on Windows sometimes hung in test_serve_noise.py after
# test_regression.py (reproduced with those two files): the gRPC aio poller thread, shared by
# the process, was gone after an earlier test's event loop closed. One file per process makes
# that far less likely; it was never seen within one file. tests/test_run_tests_script.py keeps
# this list complete.
GRPC_SERVER_TESTS="tests/test_force_language.py tests/test_noise_grpc.py tests/test_regression.py tests/test_result_log.py tests/test_serve_noise.py tests/test_speech_grpc.py tests/test_stt_denoise.py tests/test_word_pacer.py"
# The same for tools/eval/tests (paths relative to tools/eval).
EVAL_GRPC_SERVER_TESTS="tests/test_dual_wer.py tests/test_eval_wav_grpc.py tests/test_vram_and_clean.py"

status=0
ran=0
note() {   # $1 = exit status of one pytest run; 5 = no test collected (a -k that matches none here)
  if [ "$1" -eq 5 ]; then return; fi
  ran=1
  if [ "$1" -ne 0 ] && [ "$status" -eq 0 ]; then status=$1; fi
}

ignore=()
for name in $GRPC_SERVER_TESTS; do
  ignore+=("--ignore=$name")
done
echo "== server tests =="
rc=0; "$PYTHON" -m pytest tests -q "${ignore[@]}" "$@" || rc=$?; note "$rc"
for name in $GRPC_SERVER_TESTS; do
  echo "== server tests: $name =="
  rc=0; "$PYTHON" -m pytest "$name" -q "$@" || rc=$?; note "$rc"
done
eval_ignore=()
for name in $EVAL_GRPC_SERVER_TESTS; do
  eval_ignore+=("--ignore=$name")
done
echo "== eval client tests =="
rc=0; (cd "$ROOT/tools/eval" && "$PYTHON" -m pytest tests -q "${eval_ignore[@]}" "$@") || rc=$?; note "$rc"
for name in $EVAL_GRPC_SERVER_TESTS; do
  echo "== eval client tests: $name =="
  rc=0; (cd "$ROOT/tools/eval" && "$PYTHON" -m pytest "$name" -q "$@") || rc=$?; note "$rc"
done

if [ "$ran" -eq 0 ]; then
  exit 5
fi
exit "$status"
