"""Conservative punctuation-only restoration for local HF token classifier.

ASR words never change: the classifier inserts punctuation at word boundaries.
No network operations; the checkpoint and tokenizer must exist locally.
"""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class PunctResult:
    text: str
    last_mark: str = ''
    last_mark_confidence: float = 0.0


class IdentityPunctuator:
    """Explicit test-only fallback; does not synthesize punctuation."""
    def restore(self, text, language=None):
        text = text or ''
        mark = text.rstrip()[-1:] if text.rstrip().endswith(('.', '?', '!')) else ''
        return PunctResult(text, mark, 1.0 if mark else 0.0)


class HFPunctuator:
    VALID = {'.', ',', '?', ':', '-', '!'}

    def __init__(self, model_path, device='cpu', threshold=0.55):
        from pathlib import Path
        if not Path(model_path).is_dir():
            raise FileNotFoundError(f'local punctuation model directory missing: {model_path}')
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        import torch
        self.torch = torch
        self.threshold = float(threshold)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True,
                                                       use_fast=True)
        if not self.tokenizer.is_fast:
            raise ValueError('fast tokenizer required for offset mapping')
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path, local_files_only=True).to(device).eval()
        self.device = device
        self.id2label = {int(k): str(v) for k, v in self.model.config.id2label.items()}
        self.max_tokens = min(512, int(getattr(self.model.config, 'max_position_embeddings', 512)))

    def restore(self, text, language=None):
        """Insert punctuation into `text`.

        `language` is the ASR language of the utterance as the server knows it
        (for example 'Vietnamese'), or None. It is accepted so that a model or a
        threshold can later be chosen per language; this multilingual model
        does not use it yet.
        """
        text = text or ''
        words = list(re.finditer(r'\S+', text))
        if not words:
            return PunctResult(text)
        # ASR utterances are capped by server (normally ~15 seconds).
        # Never silently punctuate truncated long text.
        encoding = self.tokenizer(text, return_offsets_mapping=True, truncation=True,
                                  max_length=self.max_tokens, return_tensors='pt')
        offsets = encoding.pop('offset_mapping')[0].tolist()
        attention = encoding['attention_mask'][0].tolist()
        if attention[-1] == 1 and offsets[-1] != [0, 0]:
            # Most BERT tokenizers end with a special token, while models vary.
            # Last word must be covered for final punctuation reliability.
            pass
        max_covered = max((en for (st, en), att in zip(offsets, attention)
                           if att and en > st), default=0)
        if max_covered < words[-1].end():
            return PunctResult(text)  # safe fallback, rather than truncated output
        inputs = {k: v.to(self.device) for k, v in encoding.items()}
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits[0]
            probs = self.torch.softmax(logits, dim=-1)
            scores, ids = probs.max(dim=-1)
        labels = [self.id2label.get(int(i), '0') for i in ids.tolist()]
        confs = scores.cpu().tolist()
        spans = [(st, en) for st, en in offsets]
        cursor = 0
        pieces = []
        last_mark = ''
        last_conf = 0.0
        for w in words:
            pieces.append(text[cursor:w.end()])
            cursor = w.end()
            candidate = None
            for j, ((st, en), att) in enumerate(zip(spans, attention)):
                if att and en > st and st < w.end() and en <= w.end() and en > w.start():
                    candidate = j
            word = w.group()
            if word.endswith(('.', ',', '?', '!', ':', '-', ';')):
                # Preserve punctuation already emitted by ASR without duplication.
                if w is words[-1] and word[-1] in '.?!':
                    last_mark, last_conf = word[-1], 1.0
                continue
            if candidate is not None:
                label = labels[candidate]
                confidence = float(confs[candidate])
                if label in self.VALID and confidence >= self.threshold:
                    pieces.append(label)
                    if w is words[-1] and label in '.?!':
                        last_mark, last_conf = label, confidence
            # No changes to letters, numbers or whitespace.
        pieces.append(text[cursor:])
        return PunctResult(''.join(pieces), last_mark, last_conf)
