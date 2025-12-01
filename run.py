# main.py
import pandas as pd
from src.dataset import MultimodalDataset
from src.utils import train
from src.config import Config


def main():
    cfg = Config()

    dishes = pd.read_csv("data/dish.csv")
    ingredients = pd.read_csv("data/ingredients.csv")

    train_df = dishes[dishes["split"] == "train"].reset_index(drop=True)
    test_df = dishes[dishes["split"] == "test"].reset_index(drop=True)

    # ДОБАВЛЯЕМ нормализацию цели
    cfg.CAL_MEAN = train_df["total_calories"].mean()
    cfg.CAL_STD = train_df["total_calories"].std()

    train_ds = MultimodalDataset(train_df, ingredients, cfg, ds_type="train")
    val_ds = MultimodalDataset(test_df, ingredients, cfg, ds_type="test")

    train(cfg, train_ds, val_ds)


if __name__ == "__main__":
    main()
