"""Character-level tokenizer.

Deliberately the dumbest tokenizer that works. The model doesn't get to know
what a "word" is — it has to figure that out from scratch, which makes the
early failure modes much funnier.

Upgrade path is a byte-level BPE later; the interface here matches so the
swap is a one-liner.
"""

import json
from pathlib import Path


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str):
        # silently drop characters we've never seen; the model can't use them anyway
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path):
        Path(path).write_text(
            json.dumps({"chars": self.chars}, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["chars"])
