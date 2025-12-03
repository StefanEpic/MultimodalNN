# src/config.py
class Config:
    # Модели
    TEXT_MODEL_NAME = "bert-base-uncased"
    IMAGE_MODEL_NAME = "tf_efficientnet_b4"

    # Гиперпараметры
    BATCH_SIZE = 32
    EPOCHS = 30
    HIDDEN_DIM = 1024
    DROPOUT = 0.2
    TEXT_LR = 1e-5  # LR для текстовой модели
    IMAGE_LR = 1e-4  # LR для изображений
    HEAD_LR = 1e-3  # LR для классификатора
    SEED = 42

    # Путь сохранения лучшей модели
    SAVE_PATH = "best_model.pth"
