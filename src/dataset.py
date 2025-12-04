# src/dataset.py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
import timm

class MultimodalDataset(Dataset):
    def __init__(self, dishes_df, ingredients_df, config, ds_type="train", target_mean=None, target_std=None, mass_mean=None, mass_std=None, use_log_target=False):
        self.dishes = dishes_df.reset_index(drop=True)
        self.ingredients = ingredients_df
        self.config = config
        self.transforms = get_transforms(config, ds_type)
        self._fix_names()
        # Нормализация таргета: используем переданные статистики (возможно в лог-шкале)
        self.mean = target_mean
        self.std = target_std
        # Нормализация массы
        self.mass_mean = mass_mean
        self.mass_std = mass_std
        self.use_log_target = use_log_target

    def _fix_names(self):
        id_to_name = dict(zip(self.ingredients["id"], self.ingredients["ingr"]))
        def map_ingrs(s):
            if pd.isna(s):
                return []
            ids = [int(i.replace("ingr_", "")) for i in s.split(";")]
            return [id_to_name.get(i, i) for i in ids]
        self.dishes["ingredients"] = self.dishes["ingredients"].apply(map_ingrs)

    def __len__(self):
        return len(self.dishes)

    def __getitem__(self, idx):
        row = self.dishes.iloc[idx]
        ingredients = ", ".join([i for i in row['ingredients']])
        text = f"List of ingredients: {ingredients}\nTotal mass: {row['total_mass']} grams."
        image_path = f"data/images/{row['dish_id']}/rgb.png"
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        aug = self.transforms(image=img_np)
        img_tensor = aug["image"]
        raw_label = row["total_calories"]
        mass = float(row["total_mass"])

        # Нормализуем
        if self.mean and self.std and self.use_log_target:
            if self.use_log_target:
                lab = np.log1p(max(0.0, raw_label))
            else:
                lab = float(raw_label)
            label_norm = (lab - self.mean) / (self.std + 1e-9)
            mass_norm = (mass - (self.mass_mean if self.mass_mean is not None else 0.0)) / (
                        (self.mass_std if self.mass_std is not None else 1.0) + 1e-9)
        else:
            label_norm = raw_label
            mass_norm = mass

        return {
            "text": text,
            "image": img_tensor,
            "label": torch.tensor(label_norm, dtype=torch.float32),
            "mass": torch.tensor(mass_norm, dtype=torch.float32)
        }

def get_transforms(config, ds_type="train"):
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
    max_side = max(cfg.input_size[1], cfg.input_size[2])
    if ds_type == "train":
        return A.Compose([
            A.SmallestMaxSize(max_size=max_side, p=1.0),
            A.CenterCrop(height=cfg.input_size[1], width=cfg.input_size[2]),
            A.HorizontalFlip(p=0.1),
            A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15), p=0.6),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.6),
            A.RandomBrightnessContrast(p=0.1),
            A.HueSaturationValue(p=0.1),
            A.Normalize(mean=cfg.mean, std=cfg.std),
            A.pytorch.transforms.ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.SmallestMaxSize(max_size=max_side, p=1.0),
            A.CenterCrop(height=cfg.input_size[1], width=cfg.input_size[2]),
            A.Normalize(mean=cfg.mean, std=cfg.std),
            A.pytorch.transforms.ToTensorV2(),
        ])

def collate_fn(batch, tokenizer, max_length=128):
    texts = [item["text"] for item in batch]
    images = torch.stack([item["image"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    masses = torch.stack([item["mass"] for item in batch])
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
        "mass": masses,
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"]
    }
