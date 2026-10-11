from __future__ import annotations

__lazy_modules__ = [
    "huggingface_hub",
    "numpy",
    "torch",
]

from collections.abc import Sequence
from typing import Literal, overload

import numpy as np
import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
    pad_sequence,
)

from bgm_tagger._utils import Tokenizer, bio2xml

__all__ = ["BiLSTM", "Tokenizer"]


class BiLSTM(
    nn.Module,
    PyTorchModelHubMixin,
):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.25,
        *,
        labels: list[str],
        vocab: dict[str, int],
    ) -> None:
        super().__init__()
        self.char_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

        self.labels = labels
        self.tokenizer = Tokenizer(vocab)

    def encode(self, characters: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.char_embedding(characters))
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        encoded, _ = self.encoder(packed)
        encoded, _ = pad_packed_sequence(
            encoded, batch_first=True, total_length=characters.size(1)
        )
        return encoded

    def forward(self, characters: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.encode(characters, lengths)))

    def decode(
        self,
        logits: torch.Tensor,
        lengths: torch.Tensor,
        labels: Sequence[str],
    ) -> list[list[str]]:
        emissions = logits.detach().cpu().numpy()
        sequence_lengths = lengths.cpu().numpy()
        transitions = np.zeros((len(labels), len(labels)), dtype=np.float32)
        start_scores = np.zeros(len(labels), dtype=np.float32)
        for current, tag in enumerate(labels):
            if tag.startswith("I-"):
                transitions[:, current] = -np.inf
                transitions[labels.index("B-" + tag[2:]), current] = 0
                transitions[current, current] = 0
                start_scores[current] = -np.inf

        scores = emissions[:, 0] + start_scores
        history = []
        for position in range(1, emissions.shape[1]):
            candidates = scores[:, :, None] + transitions[None, :, :]
            history.append(candidates.argmax(axis=1))
            updated = candidates.max(axis=1) + emissions[:, position]
            scores = np.where((sequence_lengths > position)[:, None], updated, scores)

        predictions = []
        for index, length in enumerate(sequence_lengths):
            last = int(scores[index].argmax())
            path = [last]
            for position in range(int(length) - 2, -1, -1):
                last = int(history[position][index, last])
                path.append(last)
            predictions.append([labels[tag_id] for tag_id in reversed(path)])
        return predictions

    @overload
    def predict(
        self,
        *text: str,
        return_type: Literal["xml"] = "xml",
    ) -> list[str]: ...

    @overload
    def predict(
        self,
        *text: str,
        return_type: Literal["bio"],
    ) -> list[list[str]]: ...

    @torch.inference_mode()
    def predict(
        self,
        *text: str,
        return_type: Literal["xml", "bio"] = "xml",
    ) -> list[str] | list[list[str]]:
        device = next(self.parameters()).device

        bio_results = []
        nonempty = [i for i, _ in enumerate(text)]

        for start in range(0, len(text), 128):
            indices = nonempty[start : start + 128]

            sequences = [
                torch.tensor(
                    self.tokenizer.encode_chars(text[i]),
                    dtype=torch.long,
                ).to(device)
                for i in indices
            ]

            lengths = torch.tensor(
                [len(sequence) for sequence in sequences],
                dtype=torch.long,
            ).to(device)

            characters = pad_sequence(
                sequences,
                batch_first=True,
                padding_value=0,
            ).to(device)

            bio_results.extend(
                self.decode(self(characters, lengths), lengths, self.labels)
            )

        match return_type:
            case "bio":
                return bio_results
            case "xml":
                return [
                    bio2xml(s, tags) for s, tags in zip(text, bio_results, strict=True)
                ]
