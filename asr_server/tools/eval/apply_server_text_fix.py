"""Patch existing Qwen gRPC server.py to clean output before punctuation/COMMIT.

Run on a copy or use the automatic timestamp backup. Does not edit
Qwen's ASRStreamingState._raw_decoded (needed for prefix rollback).
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path
import re
import shutil


def patch(path: Path) -> Path:
    original = path.read_text(encoding='utf-8')
    patched = original
    helper = 'from asr_text_clean import strip_asr_markers'
    if helper not in patched:
        anchor = 'from vad import WebRTCEndpointer'
        if anchor not in patched:
            raise ValueError('Cannot find VAD server import anchor; patch manually per README')
        patched = patched.replace(anchor, anchor + '\n' + helper, 1)

    substitutions = {
        r"\bnew_text\s*=\s*asr_state\.text\s+or\s+(['\"])\1":
            'new_text = strip_asr_markers(asr_state.text or "")',
        r"\braw\s*=\s*asr_state\.text\s+or\s+(['\"])\1":
            'raw = strip_asr_markers(asr_state.text or "")',
    }
    for pattern, replacement in substitutions.items():
        patched, count = re.subn(pattern, replacement, patched)
        if count == 0 and replacement not in patched:
            raise ValueError(f'Missing expected code in {path}; no files modified: {pattern}')
    if patched == original:
        raise ValueError('Server already patched: no changes needed')
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = path.with_name(path.name + '.' + timestamp + '.bak')
    shutil.copy2(path, backup)
    helper_src = Path(__file__).with_name('asr_text_clean.py')
    helper_dest = path.with_name('asr_text_clean.py')
    if helper_dest.exists() and helper_dest.read_bytes() != helper_src.read_bytes():
        raise FileExistsError(f'Existing {helper_dest} differs; merge carefully. Backup created at {backup}')
    shutil.copy2(helper_src, helper_dest)
    path.write_text(patched, encoding='utf-8')
    return backup


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('server_py', type=Path, help='Path to existing server.py')
    options = ap.parse_args()
    print('Original backed up to:', patch(options.server_py))
