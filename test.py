import torch

from src.dataset import MultimodalDataset
from src.model import MultimodalModel
from transformers import AutoTokenizer
from PIL import Image
import torchvision.transforms as T


def test_single_prediction(config, image_path, text, model_path):
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
    # Калорий: 330
    print(f"Текст: {text}")
    print(f"Изображение: {image_path}")
    print(f"Предсказание: {pred.item():.2f}")
    print(f"Правильный ответ: 330")
    return pred.item()


# Использование
if __name__ == "__main__":
    from src.config import Config
    import pandas as pd
    import random

    config = Config()

    # Загружаем данные
    dishes = pd.read_csv("data/dish.csv")
    ingredients = pd.read_csv("data/ingredients.csv")

    # Создаем тестовый датасет
    test_df = dishes[dishes["split"] == "test"].reset_index(drop=True)
    test_ds = MultimodalDataset(test_df, ingredients, config, ds_type="test")

    # Выбираем 5 случайных индексов
    random_indices = random.sample(range(len(test_ds)), min(5, len(test_ds)))

    print("Предсказания для 5 случайных блюд из тестового набора:")
    print("=" * 60)

    for i, idx in enumerate(random_indices):
        # Получаем элемент из датасета
        item = test_ds[idx]

        # Получаем соответствующую строку из DataFrame для истинных значений
        row = test_df.iloc[idx]

        # Формируем путь к изображению
        image_path = f"data/images/{row['dish_id']}/rgb.png"

        # Получаем предсказание
        prediction = test_single_prediction(
            config=config,
            image_path=image_path,
            text=item["text"],
            model_path="best_model.pth"
        )

        # Выводим сравнение с истинным значением
        true_calories = row['total_calories']
        error = abs(prediction - true_calories)
        print(f"Ошибка предсказания: {error:.2f} калорий")
        print("=" * 60)
