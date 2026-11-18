"""Rebuild speech_pb2.py from src/asr_server/proto/speech.proto.

Development machine only: needs grpcio-tools (python -m pip install grpcio-tools).
The GPU server does not need protoc or grpcio-tools; it uses the files this script
writes. The generated module embeds the serialized descriptor and has no
protobuf/grpcio version checks, so it loads on any protobuf >= 4.25.

Writes two identical copies:
  src/asr_server/proto/speech_pb2.py     used by the server
  tools/eval/speech_pb2.py               used by the standalone evaluation client
speech_pb2_grpc.py is written by hand; update it if the service methods change.
"""
from pathlib import Path
import subprocess
import sys
import tempfile

from google.protobuf import descriptor_pb2

ROOT = Path(__file__).resolve().parents[1]
PROTO_DIR = ROOT / 'src' / 'asr_server' / 'proto'
TARGETS = [PROTO_DIR / 'speech_pb2.py', ROOT / 'tools' / 'eval' / 'speech_pb2.py']

HEADER = '''# Generated from speech.proto by scripts/build_proto.py. Do not edit directly.
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf.internal import builder as _builder
DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile('''
TAIL = ''')
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'speech_pb2', globals())
'''


def main():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / 'speech.desc'
        subprocess.run([sys.executable, '-m', 'grpc_tools.protoc', f'-I{PROTO_DIR}',
                        f'--descriptor_set_out={out}', 'speech.proto'], check=True)
        files = descriptor_pb2.FileDescriptorSet.FromString(out.read_bytes()).file
    assert len(files) == 1 and files[0].name == 'speech.proto'
    files[0].ClearField('source_code_info')
    text = HEADER + repr(files[0].SerializeToString(deterministic=True)) + TAIL
    for target in TARGETS:
        target.write_text(text, encoding='utf-8')
        print('wrote', target)


if __name__ == '__main__':
    main()
