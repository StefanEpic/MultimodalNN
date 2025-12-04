import numpy as np
import torch

from src.dataset import MultimodalDataset
from src.model import MultimodalModel
from transformers import AutoTokenizer
from PIL import Image
import torchvision.transforms as T    
from src.config import Config
import pandas as pd
import random


def test_single_prediction(config, image_path, text, model_path, target_std, target_mean):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Загружаем модель
    model = MultimodalModel(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Загружаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    # Подготовка изображения
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Подготовка текста
    text_encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # Предсказание
    with torch.no_grad():
        pred = model(
            input_ids=text_encoded['input_ids'].to(device),
            attention_mask=text_encoded['attention_mask'].to(device),
            image=image_tensor
        )
    pred = pred * target_std + target_mean
    pred = torch.expm1(pred).clamp(min=0.0)

    print(f"Текст: {text}")
    print(f"Изображение: {image_path}")
    print(f"Предсказание: {pred.item():.2f}")
    return pred.item()



if __name__ == "__main__":
    cfg = Config()

    dishes = pd.read_csv("data/dish.csv")
    ingredients = pd.read_csv("data/ingredients.csv")

    test_df = dishes[dishes["split"] == "test"].reset_index(drop=True)

    test_ds = MultimodalDataset(test_df, ingredients, cfg, ds_type="test")
    train_targets = np.log1p(test_df["total_calories"].clip(lower=0).values)
    test_mean = train_targets.mean()
    test_std = train_targets.std()

    # Выбираем 5 случайных индексов
    random_indices = random.sample(range(len(test_ds)), min(5, len(test_ds)))

    print("Предсказания для 5 случайных блюд из тестового набора:")
    print("=" * 60)

    mae_errors = []
    for i, idx in enumerate(random_indices):
        # Получаем элемент из датасета
        item = test_ds[idx]

        # Получаем соответствующую строку из DataFrame для истинных значений
        row = test_df.iloc[idx]

        # Формируем путь к изображению
        image_path = f"data/images/{row['dish_id']}/rgb.png"
        
        # Получаем предсказание
        prediction = test_single_prediction(
            config=cfg,
            image_path=image_path,
            text=item["text"],
            model_path="best_model.pth", 
            target_std=test_std,
            target_mean=test_mean,
        )

        # Выводим сравнение с истинным значением
        true_calories = row['total_calories']
        error = abs(prediction - true_calories)
        mae_errors.append(error)
        print(f"Правильный ответ: {true_calories:.2f}")
        print(f"Ошибка предсказания: {error:.2f} калорий")
        print("=" * 60)

    final_mae = sum(mae_errors) / len(mae_errors)
    print(f"\nСредний MAE по {len(mae_errors)} примерам: {final_mae:.2f} калорий")