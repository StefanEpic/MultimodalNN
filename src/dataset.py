# src/dataset.py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from PIL import Image
import albumentations as A
import timm


class MultimodalDataset(Dataset):
    def __init__(self, dishes_df, ingredients_df, config, ds_type="train"):
        self.dishes = dishes_df.reset_index(drop=True)
        self.ingredients = ingredients_df
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
        self.transforms = get_transforms(config, ds_type)
        # Сливаем csv
        self._fix_names()
        # Нормализация таргета
        self.mean = dishes_df["total_calories"].mean()
        self.std = dishes_df["total_calories"].std()


    def _fix_names(self):
        id_to_name = dict(zip(self.ingredients["id"], self.ingredients["ingr"]))
        def map_ingrs(s):
            if pd.isna(s):
                return []
            ids = s.split(";")
            return [id_to_name.get(i, i) for i in ids]
        self.dishes["ingredients"] = self.dishes["ingredients"].apply(map_ingrs)

    def __len__(self):
        return len(self.dishes)

    def __getitem__(self, idx):
        row = self.dishes.iloc[idx]
        ingredients = ", ".join([i for i in row['ingredients']])
        text = f"List of ingredients: {ingredients}\nTotal mass: {row['total_mass']} grams."
        image_path = f"data/images/{row['dish_id']}/rgb.png"
        try:
            img = Image.open(image_path).convert("RGB")
        except:
            img = Image.new("RGB", (256, 256), "white")
        img_np = np.array(img)
        aug = self.transforms(image=img_np)
        img_tensor = aug["image"]
        label_norm = (row["total_calories"] - self.mean) / self.std
        return {
            "text": text,
            "image": img_tensor,
            "label": torch.tensor(label_norm, dtype=torch.float32)
        }


def get_transforms(config, ds_type="train"):
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
    if ds_type == "train":
        return A.Compose([
            A.SmallestMaxSize(max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
            # A.RandomCrop(height=cfg.input_size[1], width=cfg.input_size[2]),
            # Заменяем на центральную вырезку, блюда в центре
            A.CenterCrop(height=cfg.input_size[1], width=cfg.input_size[2]),
            A.HorizontalFlip(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.7),
            # Добавляем цвета
            A.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.1, hue=0.1, p=0.6),
            A.RandomBrightnessContrast(p=0.2),
            A.HueSaturationValue(p=0.2),
            # Нормализуем
            A.Normalize(mean=cfg.mean, std=cfg.std),
            A.pytorch.transforms.ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.SmallestMaxSize(max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
            A.CenterCrop(height=cfg.input_size[1], width=cfg.input_size[2]),
            A.Normalize(mean=cfg.mean, std=cfg.std),
            A.pytorch.transforms.ToTensorV2(),
        ])


def collate_fn(batch, tokenizer, max_length=128):
    texts = [item["text"] for item in batch]
    images = torch.stack([item["image"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    tok = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    return {
        "image": images,
        "label": labels,
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"]
    }
