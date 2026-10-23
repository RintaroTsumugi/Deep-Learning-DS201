import os
os.makedirs("./checkpoints/assignment-4", exist_ok=True)

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm

from models.resnet50 import ResNet50

device = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_DIR = "./VinaFood21/VinaFood21/train"
TEST_DIR = "./VinaFood21/VinaFood21/test"

BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 10

def evaluate(dataloader: DataLoader, model: nn.Module) -> dict:
    model.eval()
    predictions = []
    trues = []
    with torch.no_grad():
        for image, label in tqdm(dataloader, desc="Evaluating"):
            image = image.to(device)
            output = model(image).argmax(dim=-1)

            predictions.extend(output.cpu().tolist())
            trues.extend(label.tolist())

    return {
        "accuracy": accuracy_score(trues, predictions),
        "precision": precision_score(trues, predictions, average="macro", zero_division=0),
        "recall": recall_score(trues, predictions, average="macro", zero_division=0),
        "f1": f1_score(trues, predictions, average="macro", zero_division=0),
    }


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # ImageFolder: mỗi lớp là một thư mục con bên trong train/ và test/
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    print("Số lớp:", len(train_dataset.classes), "| train:", len(train_dataset), "| test:", len(test_dataset))
    assert len(train_dataset.classes) == 21, "Cấu trúc thư mục không như mong đợi, kiểm tra lại"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    model = ResNet50(num_classes=21).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_score = 0
    best_score_name = "f1"
    for epoch in range(EPOCHS):
        losses = []
        model.train()
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1} - Training") as pb:
            for image, label in pb:
                image = image.to(device)
                label = label.to(device)

                # forward pass
                output = model(image)
                loss = loss_fn(output, label)
                losses.append(loss.item())

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pb.set_postfix({"loss": np.array(losses).mean()})

        scores = evaluate(test_dataloader, model)
        for score in scores:
            print(f"\t- {score}: {scores[score]}")

        current_score = scores[best_score_name]
        if current_score > best_score:
            best_score = current_score
            torch.save(
                model.state_dict(),
                "./checkpoints/assignment-4/best_model.pth"
            )