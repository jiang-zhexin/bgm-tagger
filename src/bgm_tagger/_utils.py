from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Sequence
from typing import Self
from xml.sax.saxutils import escape

__all__ = ["bio2xml", "xml2bio"]


class Tokenizer:
    def __init__(self, vocab: dict[str, int]) -> None:
        self.vocab = vocab

    @classmethod
    def fit(cls, texts: Sequence[str], max_vocab: int = 8192) -> Self:
        counts = Counter(c for text in texts for c in text)
        chars = sorted(counts, key=lambda c: (-counts[c], c))[: max_vocab - 2]
        return cls({c: i + 2 for i, c in enumerate(chars)})

    def encode_chars(self, text: str) -> list[int]:
        return [self.vocab.get(c, 1) for c in text]

    @property
    def size(self) -> int:
        return len(self.vocab) + 2


def bio2xml(text: str, tags: Sequence[str]) -> str:
    if len(text) != len(tags):
        raise ValueError(
            "The number of text characters must match the number of BIO tags."
        )

    parts: list[str] = []
    entity_type: str | None = None
    start = 0

    for position in range(len(tags) + 1):
        tag = tags[position] if position < len(tags) else "O"
        if tag.startswith("I-") and tag[2:] == entity_type:
            continue

        if entity_type is not None:
            parts.append(f"<{entity_type}>")
            parts.append(escape(text[start:position]))
            parts.append(f"</{entity_type}>")
        else:
            parts.append(escape(text[start:position]))

        entity_type = None if tag == "O" else tag[2:]
        start = position

    return "".join(parts)


def xml2bio(xml: str) -> tuple[str, list[str]]:
    root = ET.fromstring("<root>" + xml + "</root>")

    bio = ["O"] * len(root.text or "")

    for entity in root:
        if entity.attrib or len(entity):
            raise ValueError(
                f"BIO does not support attributes or nested entities.：{entity.tag}"
            )

        bio.extend(
            [f"B-{entity.tag}"] + [f"I-{entity.tag}"] * (len(entity.text or "") - 1)
        )
        if l := entity.tail:
            bio.extend(["O"] * len(l))

    return "".join(root.itertext()), bio
