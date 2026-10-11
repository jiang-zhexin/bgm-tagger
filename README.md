# bgm-tagger

A tiny character-level BiLSTM sequence tagger that parses anime release
filenames into BIO/XML entities.

## Installation

```bash
uv add bgm-tagger
# or
pip install bgm-tagger
```

## Quick start

```python
from bgm_tagger import BiLSTM

model = BiLSTM.from_pretrained("jiangzhexin/bgm-tagger-bilstm")
model.eval()

# Move to GPU if available
model.to("cuda")

text = "【喵萌奶茶屋】★01月新番★[迷宫饭 / Dungeon Meshi / Delicious in Dungeon][12][1080p][简日双语][招募翻译时轴]"

# One XML fragment per input filename
print(*model.predict(text), sep="\n")

# Or raw BIO tag sequences
print(*model.predict(text, return_type="bio"), sep="\n")
```

## Pretrained model

Pretrained weights are hosted on Hugging Face:

- Model: [jiangzhexin/bgm-tagger-bilstm](https://huggingface.co/jiangzhexin/bgm-tagger-bilstm)
- Training dataset: [jiangzhexin/bgm-bio-xml](https://huggingface.co/datasets/jiangzhexin/bgm-bio-xml)
