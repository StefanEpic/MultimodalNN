import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def analyze_target_distribution(dishes_df, ingredients_df=None, show_plots=True):
    """
    Анализ распределения калорийности блюд и выявление выбросов
    """
    df = dishes_df.copy()

    print("=" * 60)
    print("АНАЛИЗ РАСПРЕДЕЛЕНИЯ КАЛОРИЙНОСТИ")
    print("=" * 60)

    # Базовая статистика
    print("\n📊 БАЗОВАЯ СТАТИСТИКА:")
    print(f"Количество образцов: {len(df)}")
    print(f"Минимальная калорийность: {df['total_calories'].min():.2f}")
    print(f"Средняя калорийность: {df['total_calories'].mean():.2f}")
    print(f"Медианная калорийность: {df['total_calories'].median():.2f}")
    print(f"Максимальная калорийность: {df['total_calories'].max():.2f}")
    print(f"Стандартное отклонение: {df['total_calories'].std():.2f}")
    print(f"Коэффициент вариации: {(df['total_calories'].std() / df['total_calories'].mean() * 100):.2f}%")

    # Квартили и IQR
    Q1 = df['total_calories'].quantile(0.25)
    Q3 = df['total_calories'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print(f"\n📐 ГРАНИЦЫ ВЫБРОСОВ (метод IQR):")
    print(f"Q1 (25% перцентиль): {Q1:.2f}")
    print(f"Q3 (75% перцентиль): {Q3:.2f}")
    print(f"IQR: {IQR:.2f}")
    print(f"Нижняя граница: {lower_bound:.2f}")
    print(f"Верхняя граница: {upper_bound:.2f}")

    # Выбросы
    outliers = df[(df['total_calories'] < lower_bound) | (df['total_calories'] > upper_bound)]
    print(f"\n⚠️  ВЫБРОСЫ:")
    print(f"Количество выбросов: {len(outliers)} ({len(outliers) / len(df) * 100:.2f}%)")
    print(f"Минимальный выброс: {outliers['total_calories'].min() if len(outliers) > 0 else 'N/A'}")
    print(f"Максимальный выброс: {outliers['total_calories'].max() if len(outliers) > 0 else 'N/A'}")

    # Проверка на экстремальные значения
    extreme_threshold = 3000
    extreme_values = df[df['total_calories'] > extreme_threshold]
    print(f"\n🔥 ЭКСТРЕМАЛЬНЫЕ ЗНАЧЕНИЯ (> {extreme_threshold} калорий):")
    print(f"Количество: {len(extreme_values)}")
    if len(extreme_values) > 0:
        print("Примеры экстремальных значений:")
        for idx, row in extreme_values.head(5).iterrows():
            print(f"  - Dish ID: {row.get('dish_id', idx)}, Calories: {row['total_calories']:.0f}")

    # Гистограммы с разными масштабами
    if show_plots:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Распределение калорийности блюд', fontsize=16)

        # 1. Исходное распределение
        axes[0, 0].hist(df['total_calories'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(df['total_calories'].mean(), color='red', linestyle='--',
                           label=f'Среднее: {df["total_calories"].mean():.1f}')
        axes[0, 0].axvline(df['total_calories'].median(), color='green', linestyle='--',
                           label=f'Медиана: {df["total_calories"].median():.1f}')
        axes[0, 0].set_xlabel('Калории')
        axes[0, 0].set_ylabel('Частота')
        axes[0, 0].set_title('Исходное распределение')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Боксплот
        axes[0, 1].boxplot(df['total_calories'], vert=False)
        axes[0, 1].set_xlabel('Калории')
        axes[0, 1].set_title('Боксплот')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. QQ-plot (проверка нормальности)
        stats.probplot(df['total_calories'], dist="norm", plot=axes[0, 2])
        axes[0, 2].set_title('QQ-plot (нормальное распределение)')
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Логарифмированное распределение
        if (df['total_calories'] > 0).all():
            log_calories = np.log1p(df['total_calories'])
            axes[1, 0].hist(log_calories, bins=50, edgecolor='black', alpha=0.7)
            axes[1, 0].set_xlabel('log(Калории + 1)')
            axes[1, 0].set_ylabel('Частота')
            axes[1, 0].set_title('Логарифмированное распределение')
            axes[1, 0].grid(True, alpha=0.3)

            # QQ-plot для логарифмированных значений
            stats.probplot(log_calories, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title('QQ-plot после лог-трансформации')
            axes[1, 1].grid(True, alpha=0.3)

        # 5. Распределение с обрезкой выбросов (95% перцентиль)
        percentile_95 = df['total_calories'].quantile(0.95)
        clipped = df[df['total_calories'] <= percentile_95]['total_calories']
        axes[1, 2].hist(clipped, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 2].set_xlabel('Калории (обрезано при 95% перцентиле)')
        axes[1, 2].set_ylabel('Частота')
        axes[1, 2].set_title(f'Распределение после обрезки (> {percentile_95:.0f})')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return {
        'df': df,
        'outliers': outliers,
        'extreme_values': extreme_values,
        'stats': {
            'mean': df['total_calories'].mean(),
            'median': df['total_calories'].median(),
            'std': df['total_calories'].std(),
            'q1': Q1,
            'q3': Q3,
            'iqr': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'percentile_95': df['total_calories'].quantile(0.95),
            'percentile_99': df['total_calories'].quantile(0.99)
        }
    }


# Пример использования в основном скрипте
if __name__ == "__main__":
    # Загрузка данных
    dishes_df = pd.read_csv("data/dish.csv")
    ingredients_df = pd.read_csv("data/ingredients.csv")

    # Анализ распределения
    analysis = analyze_target_distribution(dishes_df, ingredients_df, show_plots=True)

    print("\n" + "=" * 60)
    print("РЕКОМЕНДАЦИИ ПО ПРЕДОБРАБОТКЕ:")
    print("=" * 60)

    stats = analysis['stats']

    if len(analysis['extreme_values']) > 0:
        print(f"⚠️  Обнаружены экстремальные значения (> 3000 калорий): {len(analysis['extreme_values'])} шт.")
        print("   Рекомендации:")
        print("   1. Проверить корректность данных (ошибки ввода)")
        print("   2. Использовать лог-трансформацию (target_transform='log')")
        print("   3. Обрезать выбросы (clip_percentile=0.95 или 0.99)")

    # Проверка на skewness (асимметрию)
    skewness = dishes_df['total_calories'].skew()
    print(f"\n📈 Коэффициент асимметрии (skewness): {skewness:.2f}")

    if abs(skewness) > 1:
        print("   Распределение сильно асимметрично!")
        if skewness > 1:
            print("   Рекомендация: используйте лог-трансформацию")
        else:
            print("   Рекомендация: рассмотрите другие трансформации")

    # Проверка процентного соотношения выбросов
    outlier_percentage = len(analysis['outliers']) / len(dishes_df) * 100
    print(f"\n🎯 Процент выбросов (по IQR): {outlier_percentage:.1f}%")

    if outlier_percentage > 5:
        print("   Рекомендация: используйте робастную нормализацию (target_transform='robust')")
