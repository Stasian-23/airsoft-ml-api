"""
Обучение EfficientNet-B0 для классификации изображений по подкатегориям.

Входные данные: папка с подпапками по классам (ImageFolder-структура).
  Пример: data/raw/dataset/ak/, data/raw/dataset/pistol/, ...

Запуск:
    python -m training.train_image
    python -m training.train_image --data_dir data/raw/dataset --epochs 60

M1 Mac: автоматически использует MPS. Linux с GPU: CUDA. Иначе: CPU.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# Аугментация данных (только для обучающей выборки)
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomRotation(15),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_model(num_classes: int) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def _make_weighted_sampler(
    labels: list[int], num_classes: int
) -> WeightedRandomSampler:
    """Балансировка классов через взвешенный семплер."""
    counts = torch.zeros(num_classes)
    for lbl in labels:
        counts[lbl] += 1
    class_weights = 1.0 / counts.clamp(min=1)
    sample_weights = [class_weights[lbl].item() for lbl in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def _accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    return (outputs.argmax(dim=1) == labels).float().mean().item()


def _pbar(iterable, **kwargs):
    return tqdm(iterable, **kwargs) if tqdm else iterable


def train(
    data_dir: str = "data/raw/dataset",
    output_dir: str = "models",
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_split: float = 0.15,
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)
    device = _get_device()
    print(f"Устройство: {device}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Загрузка данных
    full_dataset = ImageFolder(data_dir, transform=TRAIN_TRANSFORM)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Классы ({num_classes}): {class_names}")

    # Стратифицированный сплит
    indices = list(range(len(full_dataset)))
    labels = [lbl for _, lbl in full_dataset.samples]
    train_idx, val_idx = train_test_split(
        indices, test_size=val_split, stratify=labels, random_state=seed
    )

    train_subset = torch.utils.data.Subset(full_dataset, train_idx)

    val_dataset = ImageFolder(data_dir, transform=VAL_TRANSFORM)
    val_subset = torch.utils.data.Subset(val_dataset, val_idx)

    # Взвешенный семплер по обучающей подвыборке
    train_labels = [labels[i] for i in train_idx]
    sampler = _make_weighted_sampler(train_labels, num_classes)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Обучение: {len(train_idx)}  Валидация: {len(val_idx)}")

    # Модель
    model = _build_model(num_classes).to(device)

    # Фаза 1: только классификатор обучаем
    for name, param in model.named_parameters():
        param.requires_grad = "classifier" in name

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = None

    best_val_acc = 0.0
    patience_counter = 0
    PATIENCE = 12
    UNFREEZE_EPOCH = 10

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Фаза 2: размораживаем все слои на UNFREEZE_EPOCH
        if epoch == UNFREEZE_EPOCH:
            print("\nРазморозка всех слоёв...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = AdamW(model.parameters(), lr=lr / 10)
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs - UNFREEZE_EPOCH, eta_min=1e-6)

        # Обучение
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for images, batch_labels in _pbar(
            train_loader, desc=f"Эпоха {epoch}/{epochs} [train]", leave=False
        ):
            images, batch_labels = images.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += _accuracy(outputs, batch_labels)

        if scheduler is not None:
            scheduler.step()

        # Валидация
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for images, batch_labels in val_loader:
                images, batch_labels = images.to(device), batch_labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, batch_labels).item()
                val_acc += _accuracy(outputs, batch_labels)

        n_train = len(train_loader)
        n_val = len(val_loader)
        elapsed = time.time() - epoch_start

        print(
            f"Эпоха {epoch:3d}/{epochs} | "
            f"train_loss={train_loss / n_train:.4f} train_acc={train_acc / n_train:.4f} | "
            f"val_loss={val_loss / n_val:.4f} val_acc={val_acc / n_val:.4f} | "
            f"time={elapsed:.1f}s"
        )

        cur_val_acc = val_acc / n_val
        if cur_val_acc > best_val_acc:
            best_val_acc = cur_val_acc
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": class_names,
                    "num_classes": num_classes,
                    "val_acc": best_val_acc,
                    "epoch": epoch,
                },
                output_path / "image_classifier.pt",
            )
            print(f"  Сохранена лучшая модель (val_acc={best_val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE and epoch >= UNFREEZE_EPOCH:
                print(f"Ранняя остановка на эпохе {epoch} (patience={PATIENCE})")
                break

    print(f"\nОбучение завершено. Лучшая val_acc: {best_val_acc:.4f}")
    print(f"Модель сохранена: {output_path / 'image_classifier.pt'}")

    with open(output_path / "image_classes.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    print(f"Классы сохранены: {output_path / 'image_classes.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение классификатора изображений EfficientNet-B0")
    parser.add_argument("--data_dir",   default="data/raw/dataset", help="Папка с подпапками по классам")
    parser.add_argument("--output_dir", default="models",           help="Куда сохранять модель")
    parser.add_argument("--epochs",     type=int,   default=60)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_split",  type=float, default=0.15)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()
    train(**vars(args))
