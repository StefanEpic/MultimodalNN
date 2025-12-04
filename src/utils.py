# src/utils.py
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from src.model import MultimodalModel
from src.dataset import collate_fn
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import torch.cuda.amp as amp


def validate(model, loader, device, mask_text=False, mask_image=False, use_log_target=False):
    model.eval()
    mae_sum = 0.0
    count = 0
    mean = loader.dataset.mean
    std = loader.dataset.std
    with torch.no_grad():
        for batch in loader:
            preds = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                image=batch["image"].to(device),
                mass=batch["mass"].to(device),
                mask_text=mask_text,
                mask_image=mask_image
            )
            labels = batch["label"].to(device)
            # денормализация в той же шкале, в которой нормировали
            preds = preds * std + mean
            labels = labels * std + mean

            if use_log_target:
                # сейчас preds и labels в log1p-шкале, преобразуем обратно для MAE
                preds_orig = torch.expm1(preds).clamp(min=0.0)
                labels_orig = torch.expm1(labels).clamp(min=0.0)
                mae_sum += torch.sum(torch.abs(preds_orig - labels_orig)).item()
                count += labels_orig.size(0)
            else:
                mae_sum += torch.sum(torch.abs(preds - labels)).item()
                count += labels.size(0)
    return mae_sum / count

def train(config, train_ds, val_ds, use_log_target=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        num_workers=2,
        pin_memory=True
    )
    model = MultimodalModel(config).to(device)

    # Фильтруем параметры по requires_grad
    # --- собрать параметры по группам ---
    text_params = []
    image_params = []
    head_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # примерно по имени разделяем
        if name.startswith("text_model"):
            text_params.append(p)
        elif name.startswith("image_model"):
            image_params.append(p)
        else:
            head_params.append(p)

    optimizer_grouped_parameters = []
    if len(text_params) > 0:
        optimizer_grouped_parameters.append({"params": text_params, "lr": config.TEXT_LR})
    if len(image_params) > 0:
        optimizer_grouped_parameters.append({"params": image_params, "lr": config.IMAGE_LR})
    if len(head_params) > 0:
        optimizer_grouped_parameters.append({"params": head_params, "lr": config.HEAD_LR})

    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=1e-3, eps=1e-8)
    total_steps = len(train_loader) * config.EPOCHS
    num_warmup_steps = int(0.2 * total_steps)  # 10% warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=len(train_loader) * config.EPOCHS
    )

    criterion = nn.SmoothL1Loss()
    scaler = amp.GradScaler()  # mixed precision

    best_mae = 1e9
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "image": batch["image"].to(device),
                "mass": batch["mass"].to(device)
            }
            labels = batch["label"].to(device)
            with amp.autocast(enabled=(device=="cuda")):
                preds = model(**inputs)
                loss = criterion(preds, labels)
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
        scheduler.step()

        val_mae = validate(model, val_loader, device, mask_text=False, mask_image=False, use_log_target=use_log_target)
        mae_no_text = validate(model, val_loader, device, mask_text=True, mask_image=False, use_log_target=use_log_target)
        mae_no_image = validate(model, val_loader, device, mask_text=False, mask_image=True, use_log_target=use_log_target)
        print(f"Эпоха {epoch+1}/{config.EPOCHS}. Train Loss: {total_loss/len(train_loader):.4f}. Общее MAE: {val_mae:.2f}. MAE для изображений: {mae_no_text:.2f}. MAE для текста: {mae_no_image:.2f}")
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), config.SAVE_PATH)
            print(f"Сохранена модель MAE: {val_mae:.2f}")
    print(f"Было достигнуто лучшее MAE: {best_mae:.2f}")
