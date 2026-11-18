"""gRPC Python service bindings for speech.proto (written by hand, no version checks)."""
import grpc

from . import speech_pb2 as speech__pb2


class SpeechStreamStub:
    def __init__(self, channel):
        self.Recognize = channel.stream_stream(
            '/speech.SpeechStream/Recognize',
            request_serializer=speech__pb2.AudioChunk.SerializeToString,
            response_deserializer=speech__pb2.RecognitionResult.FromString,
        )


class SpeechStreamServicer:
    async def Recognize(self, request_iterator, context):
        raise NotImplementedError()


def add_SpeechStreamServicer_to_server(servicer, server):
    handler = grpc.stream_stream_rpc_method_handler(
        servicer.Recognize,
        request_deserializer=speech__pb2.AudioChunk.FromString,
        response_serializer=speech__pb2.RecognitionResult.SerializeToString,
    )
    server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(
        'speech.SpeechStream', {'Recognize': handler}),))
