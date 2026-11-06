import random
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from seqeval.metrics import f1_score, classification_report   # pip install seqeval

from tqdm import tqdm
import numpy as np

from data_utils.phoner import PhoNER, Vocab, collate_fn, IGNORE_INDEX
from models.classifiers.bilstm import Encoder

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def train(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        epoch: int,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer
    ):

    running_loss = []
    model.train()

    with tqdm(dataloader, desc=f"Epoch {epoch} - Training") as pbar:
        for item in pbar:
            input_ids = item["input_ids"].to(device)
            lengths = item["lengths"]
            labels = item["labels"].to(device)              # (bs, len)

            # forward pass
            logits = model(input_ids, lengths)              # (bs, len, n_labels)
            loss = loss_fn(logits.transpose(1, 2), labels)   # (bs, n_labels, len) 

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss.append(loss.item())
            pbar.set_postfix({
                "loss": np.array(running_loss).mean()
            })
    return float(np.mean(running_loss))

@torch.no_grad()
def evaluate(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        epoch: int,
        id2label: dict
    ) -> float:
    model.eval()

    gts = []        # ground truths: list[list[str]], mỗi phần tử là nhãn của một câu
    pds = []        # predictions:   list[list[str]]
    with tqdm(dataloader, desc=f"Epoch {epoch} - Evaluating") as pbar:
        for item in pbar:
            input_ids = item["input_ids"].to(device)
            lengths = item["lengths"]
            labels = item["labels"]                         # (bs, len), còn trên CPU

            # forward pass
            logits = model(input_ids, lengths)              # (bs, len, n_labels)
            predicted = logits.argmax(dim=-1).cpu()         # (bs, len)

            for pred, gold in zip(predicted, labels):
                mask = gold != IGNORE_INDEX                 # bỏ vị trí padding
                gts.append([id2label[i] for i in gold[mask].tolist()])
                pds.append([id2label[i] for i in pred[mask].tolist()])

    f1 = f1_score(gts, pds, average="macro")                                 
    print(f"F1: {f1}")
    return f1

if __name__ == "__main__":
    print("Loading vocab ...")
    vocab = Vocab(train_path="./PhoNER/train_syllable.json")

    print("Creating datasets ...")
    train_dataset = PhoNER(
        path="./PhoNER/train_syllable.json",
        vocab=vocab
    )
    dev_dataset = PhoNER(
        path="./PhoNER/dev_syllable.json",
        vocab=vocab
    )
    test_dataset = PhoNER(
        path="./PhoNER/test_syllable.json",
        vocab=vocab
    )

    print("Creating dataloaders ...")
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )
    dev_dataloader = DataLoader(
        dataset=dev_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )

    print("Initializing model and utilities ...")
    model = Encoder(
        vocab_size=vocab.len,
        hidden_size=256,
        n_layer=5,
        n_labels=vocab.n_labels,
        padding_idx=vocab.pad_idx
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    epoch = 0
    best_f1 = 0
    patience = 0
    while True:
        epoch += 1
        train(model, train_dataloader, epoch, loss_fn, optimizer)
        f1 = evaluate(model, dev_dataloader, epoch, vocab.i2l)

        if f1 > best_f1:
            best_f1 = f1
            patience = 0
        else:
            patience += 1

        if patience == 5:
            print("Patience reached! Training completed")
            break

    test_f1 = evaluate(model, test_dataloader, epoch, vocab.i2l)
    print(f"F1-score on test set: {test_f1}")
