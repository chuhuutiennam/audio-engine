"""Entry point: python -m asr_server --config config/server.yaml"""
import argparse
import asyncio
import logging
import sys

from .config import ConfigError, check_logging_config, load_settings, setup_logging

LOG = logging.getLogger('qwen3_asr.grpc.vad')


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog='asr_server', description='Qwen3-ASR gRPC server')
    p.add_argument('--config', default='config/server.yaml', help='path to server.yaml')
    p.add_argument('--logging-config', default='config/logging.yaml',
                   help='path to logging.yaml; pass an empty string for console-only logging')
    p.add_argument('--no-console-log', action='store_true',
                   help='drop the console log handler (used when running in the background)')
    p.add_argument('--host', help='override server.host')
    p.add_argument('--port', type=int, help='override server.port')
    p.add_argument('--check-config', action='store_true',
                   help='load and validate the configuration, print it, then exit')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        settings = load_settings(args.config, {'host': args.host, 'port': args.port})
        if args.check_config:
            # Validate only: no log handlers are opened and no files are created.
            if args.logging_config:
                check_logging_config(args.logging_config)
            for name, value in vars(settings).items():
                print(f'{name} = {value!r}')
            return 0
        setup_logging(args.logging_config or None, console=not args.no_console_log)
    except ConfigError as exc:
        print(f'configuration error: {exc}', file=sys.stderr)
        return 2
    LOG.info('configuration loaded from %s', args.config)
    from .server import serve          # imported late: pulls in grpc, numpy, webrtcvad
    try:
        asyncio.run(serve(settings))
    except KeyboardInterrupt:
        LOG.info('interrupted, shutting down')
    except FileNotFoundError as exc:
        LOG.error('%s', exc)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
