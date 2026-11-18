from types import SimpleNamespace
import torch
from asr_server.punctuation import HFPunctuator


class DummyTokenizer:
    def __call__(self, text, **kwargs):
        parts = []
        p = 0
        for w in text.split():
            start = text.index(w, p)
            parts.append((start, start+len(w)))
            p = start + len(w)
        offsets = [(0,0)] + parts + [(0,0)]
        return {'input_ids':torch.arange(len(offsets))[None],
                'attention_mask':torch.ones(1,len(offsets),dtype=torch.int64),
                'offset_mapping':torch.tensor(offsets)[None]}


class DummyModel:
    def __call__(self, **kwargs):
        length = kwargs['input_ids'].shape[1]
        # 0=no punctuation, 1=comma, 2=question mark
        logits = torch.zeros((1, length, 3))
        logits[0,:,0] = 8.0
        logits[0,2,0] = 0.0
        logits[0,2,1] = 8.0
        logits[0,length-2,0] = 0.0
        logits[0,length-2,2] = 8.0
        return SimpleNamespace(logits=logits)


def test_punctuation_inserts_without_rewriting_words():
    p = HFPunctuator.__new__(HFPunctuator)
    p.torch = torch
    p.threshold = 0.55
    p.tokenizer = DummyTokenizer()
    p.model = DummyModel()
    p.device = 'cpu'
    p.id2label = {0:'0', 1:',', 2:'?'}
    p.max_tokens = 512
    out = p.restore('xin chao anh')
    assert out.text == 'xin chao, anh?'
    assert out.last_mark == '?'
    assert out.last_mark_confidence > 0.99
    assert out.text.replace(',', '').replace('?', '') == 'xin chao anh'


def test_empty_is_not_hallucinated():
    p = HFPunctuator.__new__(HFPunctuator)
    assert p.restore('').text == ''


def test_language_argument_is_accepted_and_does_not_change_the_result():
    p = HFPunctuator.__new__(HFPunctuator)
    p.torch = torch
    p.threshold = 0.55
    p.tokenizer = DummyTokenizer()
    p.model = DummyModel()
    p.device = 'cpu'
    p.id2label = {0:'0', 1:',', 2:'?'}
    p.max_tokens = 512
    assert p.restore('xin chao anh', language='Vietnamese') == p.restore('xin chao anh')
    assert p.restore('xin chao anh', language=None).text == 'xin chao, anh?'
