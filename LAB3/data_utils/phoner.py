import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

import json
from collections import Counter

# nhãn dùng để pad, phải khớp với ignore_index của CrossEntropyLoss
IGNORE_INDEX = -100


# đọc file JSONL: mỗi dòng là một câu {"words": [...], "tags": [...]}
def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    # mỗi token phải có đúng một nhãn
    for i, item in enumerate(data):
        assert len(item["words"]) == len(item["tags"]), \
            f"Số token khác số nhãn ở dòng {i} của {path}"
    return data


# gom các mẫu thành batch
def collate_fn(items: list[dict]) -> dict:
    input_ids = [item["input_ids"] for item in items]
    labels = [item["labels"] for item in items]

    lengths = torch.tensor([ids.shape[0] for ids in input_ids], dtype=torch.long)
    max_len = int(lengths.max())

    # input pad bằng 0 (<pad>), nhãn pad bằng -100 để loss bỏ qua
    input_ids = torch.stack([
        F.pad(ids, pad=(0, max_len - ids.shape[0]), mode="constant", value=0)
        for ids in input_ids
    ])
    labels = torch.stack([
        F.pad(lab, pad=(0, max_len - lab.shape[0]), mode="constant", value=IGNORE_INDEX)
        for lab in labels
    ])

    return {
        "input_ids": input_ids,     # (bs, max_len)
        "lengths": lengths,         # (bs, ) độ dài thật, chưa tính padding
        "labels": labels            # (bs, max_len)
    }


class Vocab:
    def __init__(self, train_path: str, lowercase: bool = False, min_freq: int = 1):
        self.pad = "<pad>"
        self.unk = "<unk>"
        self.lowercase = lowercase

        data = read_jsonl(train_path)
        counter = Counter()
        labels = set()

        for item in data:
            counter.update(self.preprocess_word(w) for w in item["words"])
            labels.update(item["tags"])

        # 0 = <pad>, 1 = <unk>, các từ khác bắt đầu từ 2
        self.w2i = {
            self.pad: 0,
            self.unk: 1
        }

        # word -> idx (chỉ giữ từ xuất hiện >= min_freq lần)
        all_words = sorted(w for w, c in counter.items() if c >= min_freq) 
        for idx, word in enumerate(all_words, start=2): # w2i lưu idx của mỗi từ
            self.w2i[word] = idx    # gán id liên tiếp từ 2 trở đi

        # idx -> word
        self.i2w = {idx: word for word, idx in self.w2i.items()} # i2w lưu từ có mã idx

        # label -> idx (dựng từ train, dùng chung cho dev/test)
        self.l2i = {
            label: idx for idx, label in enumerate(sorted(labels)) # l2i lưu idx của mỗi nhãn
        }

        # idx -> label
        self.i2l = {
            idx: label for label, idx in self.l2i.items() #i2l lưu nhãn có mã idx
        }

    @property
    def n_labels(self) -> int:
        return len(self.l2i)

    @property
    def len(self) -> int:
        return len(self.w2i)        # kích thước vocab -> số dòng của Embedding

    @property
    def pad_idx(self) -> int: 
        return self.w2i[self.pad] # trả về pad = 0

    # KHÔNG bỏ dấu câu và KHÔNG tách/gộp token, vì sẽ làm lệch số token với số nhãn
    def preprocess_word(self, word: str) -> str:
        return word.lower() if self.lowercase else word

    # chuyển list token thành tensor số
    def encode_sentence(self, words: list[str]) -> torch.Tensor:
        unk_id = self.w2i[self.unk]
        word_ids = [self.w2i.get(self.preprocess_word(w), unk_id) for w in words] # từ không có trong w2i (kể cả từ bị lọc bởi min_freq) đều thành <unk>

        return torch.tensor(word_ids, dtype=torch.long)

    # chuyển list nhãn thành tensor số (cùng độ dài với câu)
    def encode_labels(self, tags: list[str]) -> torch.Tensor:
        label_ids = [self.l2i[t] for t in tags]     # trả về một danh sách idx 

        return torch.tensor(label_ids, dtype=torch.long)

    # chuyển list idx -> list nhãn dạng chuỗi (dùng cho seqeval)
    def decode_labels(self, label_ids: torch.Tensor) -> list:
        return [self.i2l[idx] for idx in label_ids.tolist()] # tra i2l để đổi từng id về chuỗi nhãn


class PhoNER(Dataset):
    def __init__(self, path: str, vocab: Vocab):
        super().__init__()

        self._data = read_jsonl(path)
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        item = self._data[idx]

        input_ids = self.vocab.encode_sentence(item["words"])
        labels = self.vocab.encode_labels(item["tags"])

        return {
            "input_ids": input_ids,     # (len, )
            "labels": labels            # (len, ) cùng độ dài với input_ids
        }