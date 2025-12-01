# src/utils.py
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from src.model import MultimodalModel
from src.dataset import collate_fn
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup


def validate(model, loader, device, mean, std):
    model.eval()
    mae_sum = 0
    count = 0
    with torch.no_grad():
        for batch in loader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "image": batch["image"].to(device)
            }
            labels = batch["label"].to(device)
            preds = model(**inputs)
            # ОБРАТНАЯ ТРАНСФОРМАЦИЯ
            preds_real = preds * std + mean
            labels_real = labels * std + mean

            mae_sum += torch.sum(torch.abs(preds_real - labels_real)).item()
            count += labels.size(0)
    return mae_sum / count


def train(config, train_ds, val_ds):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    model = MultimodalModel(config).to(device)
    optimizer = AdamW([
        {"params": model.text_model.parameters(), "lr": config.TEXT_LR},
        {"params": model.image_model.parameters(), "lr": config.IMAGE_LR},
        {"params": list(model.text_proj.parameters()) +
                   list(model.image_proj.parameters()) +
                   list(model.regressor.parameters()),
         "lr": config.HEAD_LR},
    ])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=200,
        num_training_steps=len(train_loader) * config.EPOCHS
    )
    criterion = nn.L1Loss()
    best_mae = 1e9
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "image": batch["image"].to(device)
            }
            labels = batch["label"].to(device)
            preds = model(**inputs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        # Передаём mean/std
        val_mae = validate(
            model, val_loader, device,
            config.CAL_MEAN, config.CAL_STD
        )
        print(f"Эпоха {epoch+1}/{config.EPOCHS}. Train Loss: {total_loss/len(train_loader):.2f}. MAE: {val_mae:.2f}")
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), config.SAVE_PATH)
            print(f"Сохранена модель MAE: {val_mae:.2f}")
    print(f"Было достигнуто лучшее MAE: {best_mae:.2f}")
