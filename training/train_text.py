"""
Дообучение cointegrated/rubert-tiny2 для классификации текстов по 3 категориям.

Входные данные: posts.parquet с колонками Text и categoryname.

Запуск:
    python -m training.train_text
    python -m training.train_text --data_path data/raw/posts.parquet --epochs 6

M1 Mac: автоматически использует MPS. Linux с GPU: CUDA. Иначе: CPU.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

MODEL_NAME = "cointegrated/rubert-tiny2"
# 256 токенов покрывает 95% объявлений и вдвое снижает нагрузку по сравнению с 512
MAX_LEN = 256


class PostDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int = MAX_LEN) -> None:
        # padding="max_length" даёт тензоры одинаковой формы — исключает перекомпиляцию на MPS/CUDA
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(
    data_path: str = "data/raw/posts.parquet",
    output_dir: str = "models/text_classifier",
    epochs: int = 6,
    batch_size: int = 64,
    lr: float = 3e-5,
    val_split: float = 0.1,
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)
    device = _get_device()
    print(f"Устройство: {device}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Загрузка и очистка данных
    df = pd.read_parquet(data_path)
    df = df[["Text", "categoryname"]].dropna()
    df["Text"] = df["Text"].astype(str).str.strip()
    df = df[df["Text"].str.len() > 5].reset_index(drop=True)

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["categoryname"])
    num_classes = len(le.classes_)
    id2label = {i: c for i, c in enumerate(le.classes_)}
    label2id = {c: i for i, c in enumerate(le.classes_)}

    print(f"Классы ({num_classes}): {id2label}")
    print(f"Всего примеров: {len(df)}")
    print(df["categoryname"].value_counts().to_string())

    # Сохраняем маппинг до начала обучения
    with open(output_path / "labels.json", "w", encoding="utf-8") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f, ensure_ascii=False, indent=2)

    train_df, val_df = train_test_split(
        df, test_size=val_split, stratify=df["label"], random_state=seed
    )
    print(f"\nОбучение: {len(train_df)}  Валидация: {len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Токенизация обучающей выборки...")
    train_ds = PostDataset(train_df["Text"].tolist(), train_df["label"].tolist(), tokenizer)
    print("Токенизация валидационной выборки...")
    val_ds = PostDataset(val_df["Text"].tolist(), val_df["label"].tolist(), tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    total_steps  = len(train_loader) * epochs
    warmup_steps = total_steps // 10

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        it = tqdm(train_loader, desc=f"Эпоха {epoch}/{epochs} [train]") if tqdm else train_loader
        for batch in it:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels   = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss    += outputs.loss.item()
            preds          = outputs.logits.argmax(dim=-1)
            train_correct += (preds == batch_labels).sum().item()
            train_total   += len(batch_labels)

        # Валидация
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                batch_labels   = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
                val_loss     += outputs.loss.item()
                preds         = outputs.logits.argmax(dim=-1)
                val_correct  += (preds == batch_labels).sum().item()
                val_total    += len(batch_labels)

        t_acc    = train_correct / train_total
        v_acc    = val_correct   / val_total
        elapsed  = time.time() - epoch_start

        print(
            f"Эпоха {epoch}/{epochs} | "
            f"train_loss={train_loss / len(train_loader):.4f} train_acc={t_acc:.4f} | "
            f"val_loss={val_loss / len(val_loader):.4f} val_acc={v_acc:.4f} | "
            f"time={elapsed:.1f}s"
        )

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            model.save_pretrained(str(output_path))
            tokenizer.save_pretrained(str(output_path))
            print(f"  Сохранена лучшая модель (val_acc={best_val_acc:.4f})")

    print(f"\nОбучение завершено. Лучшая val_acc: {best_val_acc:.4f}")
    print(f"Модель сохранена: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Дообучение rubert-tiny2 для классификации текстов")
    parser.add_argument("--data_path",  default="data/raw/posts.parquet",    help="Путь к parquet-файлу")
    parser.add_argument("--output_dir", default="models/text_classifier",    help="Куда сохранять модель")
    parser.add_argument("--epochs",     type=int,   default=6)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=3e-5)
    parser.add_argument("--val_split",  type=float, default=0.1)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()
    train(**vars(args))
