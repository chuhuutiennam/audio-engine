"""Demo client for SpeechStream.Recognize (speech.proto).

Sends what the RTP gateway sends: PCM16 mono 16 kHz in 80 ms AudioChunk messages,
chunk_seq from 0, RTP timestamps on an 8 kHz clock, end_of_stream on the last
chunk. The engine owns VAD and answers with PARTIAL (is_final=false) and FINAL
(is_final=true) results.

Run from the project root:
    PYTHONPATH=src python scripts/client.py --wav call_audio.wav --realtime
    PYTHONPATH=src python scripts/client.py --g711 call_audio.alaw --codec pcma --realtime
"""
import argparse
import asyncio
import wave

import grpc
import numpy as np

from asr_server.audio import PCM16kConverter, decode_audio
from asr_server.proto import speech_pb2 as pb
from asr_server.proto import speech_pb2_grpc as pb_grpc

RTP_MODULO = 1 << 32
RTP_CLOCK = 8000
SAMPLE_RATE = 16000


def load_pcm16k(args):
    """Input file -> PCM16 mono 16 kHz bytes, the way the gateway prepares it."""
    if args.wav:
        with wave.open(args.wav, 'rb') as handle:
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
                raise ValueError('expect mono 16-bit PCM WAV')
            rate = handle.getframerate()
            if rate not in (8000, 16000):
                raise ValueError('expect an 8 kHz or 16 kHz WAV')
            samples = decode_audio(handle.readframes(handle.getnframes()), 3)      # PCM_S16LE
    else:
        with open(args.g711, 'rb') as handle:
            samples = decode_audio(handle.read(), 1 if args.codec == 'pcma' else 2)
        rate = 8000
    converter = PCM16kConverter(rate)       # stateful resampler over the whole stream
    samples16 = np.concatenate([converter.push(samples), converter.flush()])
    return (np.clip(samples16, -1.0, 1.0) * 32767.0).astype('<i2').tobytes()


async def requests(args, pcm):
    chunk_bytes = SAMPLE_RATE * args.frame_ms // 1000 * 2
    sent_samples = 0
    count = (len(pcm) + chunk_bytes - 1) // chunk_bytes
    for index in range(count):
        piece = pcm[index * chunk_bytes:(index + 1) * chunk_bytes]
        samples = len(piece) // 2
        yield pb.AudioChunk(
            session_id=args.session_id, stream_id=args.stream_id,
            pcm=piece, sample_rate=SAMPLE_RATE, channels=1,
            rtp_timestamp=(args.rtp_start + sent_samples * RTP_CLOCK // SAMPLE_RATE) % RTP_MODULO,
            duration_ms=samples * 1000 // SAMPLE_RATE,
            end_of_stream=(index == count - 1),
            language=args.language, task='speech_to_text',
            rtp_clock_rate=RTP_CLOCK, chunk_seq=index)
        sent_samples += samples
        if args.realtime:
            await asyncio.sleep(samples / SAMPLE_RATE)


async def main(args):
    pcm = load_pcm16k(args)
    committed = []
    async with grpc.aio.insecure_channel(args.target) as channel:
        stub = pb_grpc.SpeechStreamStub(channel)
        async for result in stub.Recognize(requests(args, pcm)):
            if result.is_final:
                committed.append(result.text)
                print(f'[FINAL   seq={result.seq} utt={result.utterance_id} '
                      f'{pb.CommitReason.Name(result.reason)} lang={result.language or "-"} '
                      f'rtp {result.ts_start}-{result.ts_end}] {result.text}', flush=True)
            else:
                # REPLACE the text shown for this utterance; empty text means "clear it".
                print(f'[PARTIAL seq={result.seq} utt={result.utterance_id}:{result.revision}] '
                      f'{result.text or "<cleared>"}', flush=True)
    print('FINAL CALL:', ' '.join(committed))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--wav', help='mono 16-bit PCM WAV, 8 or 16 kHz')
    group.add_argument('--g711', help='raw G.711 payload file (no RTP headers), 8 kHz')
    parser.add_argument('--codec', choices=['pcma', 'pcmu'], default='pcma')
    parser.add_argument('--target', default='127.0.0.1:50051')
    parser.add_argument('--session-id', default='call-test-1')
    parser.add_argument('--stream-id', default='call-test-1-a2b')
    parser.add_argument('--language', default='vi', help='vi | en | empty string for auto')
    parser.add_argument('--frame-ms', type=int, choices=[20, 40, 80, 100], default=80)
    parser.add_argument('--rtp-start', type=int, default=0)
    parser.add_argument('--realtime', action='store_true',
                        help='pace like a live call; without it the engine ingress limit applies')
    return parser.parse_args()


if __name__ == '__main__':
    asyncio.run(main(parse_args()))
