import numpy as np
import pandas as pd
import timm
import torch

import albumentations as A
from PIL import Image

from src.config import Config
from src.dataset import MultimodalDataset, collate_fn

class MultimodalDatasetUpdated(MultimodalDataset):
    def get_original_image(self, idx):
        """Получить оригинальное изображение без аугментаций"""
        row = self.dishes.iloc[idx]
        image_path = f"data/images/{row['dish_id']}/rgb.png"
        try:
            img = Image.open(image_path).convert("RGB")
        except:
            img = Image.new("RGB", (256, 256), "white")
        return img

    def get_augmented_image(self, idx, apply_transforms=True):
        """Получить изображение с аугментациями"""
        row = self.dishes.iloc[idx]
        image_path = f"data/images/{row['dish_id']}/rgb.png"
        try:
            img = Image.open(image_path).convert("RGB")
        except:
            img = Image.new("RGB", (256, 256), "white")

        img_np = np.array(img)

        if apply_transforms:
            aug = self.transforms(image=img_np)
            img_tensor = aug["image"]
        else:
            # Только ресайз и нормализация для сравнения
            cfg = timm.get_pretrained_cfg(self.config.IMAGE_MODEL_NAME)
            test_transform = A.Compose([
                A.SmallestMaxSize(max_size=max(cfg.input_size[1], cfg.input_size[2]), p=1.0),
                A.CenterCrop(height=cfg.input_size[1], width=cfg.input_size[2]),
                A.Normalize(mean=cfg.mean, std=cfg.std),
                A.pytorch.transforms.ToTensorV2(),
            ])
            aug = test_transform(image=img_np)
            img_tensor = aug["image"]

        return img_tensor

    def visualize_augmentations(self, idx, n_samples=5):
        """Визуализировать несколько аугментированных версий одного изображения"""
        fig, axes = plt.subplots(1, n_samples + 1, figsize=(20, 4))

        # Оригинальное изображение
        original_img = self.get_original_image(idx)
        axes[0].imshow(original_img)
        axes[0].set_title("Original")
        axes[0].axis('off')

        # Аугментированные версии
        for i in range(n_samples):
            img_tensor = self.get_augmented_image(idx, apply_transforms=True)
            # Конвертируем тензор обратно в изображение для отображения
            img_for_display = self.tensor_to_image(img_tensor)
            axes[i + 1].imshow(img_for_display)
            axes[i + 1].set_title(f"Augmented {i + 1}")
            axes[i + 1].axis('off')

        plt.tight_layout()
        plt.show()

    def tensor_to_image(self, tensor):
        """Конвертировать тензор обратно в изображение для отображения"""
        # Денормализация
        cfg = timm.get_pretrained_cfg(self.config.IMAGE_MODEL_NAME)
        mean = torch.tensor(cfg.mean).view(3, 1, 1)
        std = torch.tensor(cfg.std).view(3, 1, 1)

        tensor_denorm = tensor * std + mean
        tensor_denorm = torch.clamp(tensor_denorm, 0, 1)

        # Конвертируем в numpy и меняем оси
        img_np = tensor_denorm.numpy().transpose(1, 2, 0)
        return img_np


def visualize_batch(batch, n_images=4):
    """Визуализировать батч изображений"""
    images = batch['image']
    texts = batch['text']
    labels = batch['label']

    n_images = min(n_images, len(images))
    fig, axes = plt.subplots(2, n_images, figsize=(15, 6))

    for i in range(n_images):
        # Верхний ряд - изображения
        img_tensor = images[i]
        img_np = denormalize_and_convert(img_tensor)

        axes[0, i].imshow(img_np)
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Calories: {labels[i]:.1f}")

        # Нижний ряд - текст (первые 100 символов)
        text_preview = texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i]
        axes[1, i].text(0.5, 0.5, text_preview,
                        horizontalalignment='center',
                        verticalalignment='center',
                        wrap=True,
                        fontsize=8)
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()


def denormalize_and_convert(tensor, mean=None, std=None):
    """Денормализовать тензор изображения"""
    if mean is None or std is None:
        cfg = timm.get_pretrained_cfg("resnet50")  # или ваша модель
        mean = torch.tensor(cfg.mean).view(3, 1, 1)
        std = torch.tensor(cfg.std).view(3, 1, 1)

    tensor_denorm = tensor * std + mean
    tensor_denorm = torch.clamp(tensor_denorm, 0, 1)
    img_np = tensor_denorm.numpy().transpose(1, 2, 0)

    return img_np

# В вашем основном скрипте
import matplotlib.pyplot as plt

cfg = Config()

dishes = pd.read_csv("data/dish.csv")
ingredients = pd.read_csv("data/ingredients.csv")

train_df = dishes[dishes["split"] == "train"].reset_index(drop=True)
# Создаем датасет
dataset = MultimodalDatasetUpdated(train_df, ingredients, cfg, ds_type="train")

# Визуализируем аугментации для одного изображения
dataset.visualize_augmentations(idx=0, n_samples=5)

# Или получить конкретное изображение
original = dataset.get_original_image(0)
augmented = dataset.get_augmented_image(0, apply_transforms=True)

# Конвертируем тензор для отображения
augmented_img = dataset.tensor_to_image(augmented)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(original)
axes[0].set_title("Original")
axes[0].axis('off')

axes[1].imshow(augmented_img)
axes[1].set_title("Augmented")
axes[1].axis('off')

plt.show()

# Визуализация батча из DataLoader
from torch.utils.data import DataLoader

def custom_collate(batch):
    return collate_fn(batch, dataset.tokenizer, max_length=128)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)

# Получаем один батч
batch = next(iter(dataloader))

# Визуализируем
visualize_batch({
    'image': batch['image'],
    'text': [dataset.dishes.iloc[i]['ingredients'] for i in range(len(batch['image']))],
    'label': batch['label']
})