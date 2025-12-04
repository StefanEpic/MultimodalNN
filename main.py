# main.py
import pandas as pd
import numpy as np
from src.dataset import MultimodalDataset
from src.utils import train
from src.config import Config
from sklearn.model_selection import train_test_split

def main():
    cfg = Config()

    dishes = pd.read_csv("data/dish.csv")
    ingredients = pd.read_csv("data/ingredients.csv")

    # Опционально: отсечь экстремальные выбросы
    dishes = dishes[dishes["total_calories"] < 700].reset_index(drop=True)
    dishes = dishes[dishes["total_mass"] < 800]

    train_df = dishes[dishes["split"] == "train"].reset_index(drop=True)
    # test_df = dishes[dishes["split"] == "test"].reset_index(drop=True)

    # Используем log1p трансформацию таргета — это часто стабилизирует обучение при сильной скошенности
    use_log = True

    if use_log:
        train_targets = np.log1p(train_df["total_calories"].clip(lower=0).values)
    else:
        train_targets = train_df["total_calories"].values

    train_mean = train_targets.mean()
    train_std = train_targets.std()

    # Масса как дополнительная числовая фича
    mass_vals = train_df["total_mass"].values
    mass_mean = mass_vals.mean()
    mass_std = mass_vals.std()
    
    train_df, val_df = train_test_split(
        train_df, 
        test_size=0.2,  
        random_state=42,  
        shuffle=True
    )

    train_ds = MultimodalDataset(train_df, ingredients, cfg, ds_type="train",
                                 target_mean=train_mean, target_std=train_std,
                                 mass_mean=mass_mean, mass_std=mass_std,
                                 use_log_target=use_log)
    
    val_ds = MultimodalDataset(val_df, ingredients, cfg, ds_type="test",
                               target_mean=train_mean, target_std=train_std,
                               mass_mean=mass_mean, mass_std=mass_std,
                               use_log_target=use_log)

    train(cfg, train_ds, val_ds, use_log_target=use_log)

if __name__ == "__main__":
    main()
