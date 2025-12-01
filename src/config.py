# src/config.py
class Config:
    # Модели
    TEXT_MODEL_NAME = "bert-base-uncased"
    IMAGE_MODEL_NAME = "tf_efficientnet_b1"

    TEXT_MODEL_UNFREEZE = "encoder.layer.11|pooler"
    IMAGE_MODEL_UNFREEZE = "blocks.6|conv_head|bn2"

    # Гиперпараметры
    BATCH_SIZE = 16
    EPOCHS = 30
    HIDDEN_DIM = 768  # Увеличиваем пытаясь достичь MAE < 50
    DROPOUT = 0.2
    TEXT_LR = 3e-5  # LR для текстовой модели
    IMAGE_LR = 1e-4  # LR для изображений
    HEAD_LR = 5e-4  # LR для классификатора
    SEED = 42

    # Путь сохранения лучшей модели
    SAVE_PATH = "best_model.pth"
