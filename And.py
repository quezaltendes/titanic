# Шаг 1: Установка AutoGluon (если не установлен)
!pip install autogluon.multimodal

# Импорт библиотек
import pandas as pd
import numpy as np
from autogluon.multimodal import MultiModalPredictor
import os

# Константы
TRAIN_IMG_DIR = '/kaggle/input/cv-avito-car-price-prediction/train_images/train_images'
TEST_IMG_DIR = '/kaggle/input/cv-avito-car-price-prediction/test_images/test_images'
TRAIN_DATA_PATH = '/kaggle/input/cv-avito-car-price-prediction/train_dataset.parquet'
TEST_DATA_PATH = '/kaggle/input/cv-avito-car-price-prediction/test_dataset.parquet'

# Шаг 2: Подготовка данных
def prepare_data(train_df, test_df, train_img_dir, test_img_dir):
    """
    Создает DataFrame с колонками 'image' и всеми табличными признаками.
    """
    def add_image_paths(df, img_dir, id_column='id'):
        image_paths = []
        for idx in df.index:
            found = False
            for i in range(4):  # Максимум 4 изображения на объявление
                img_path = os.path.join(img_dir, f"{idx}_{i}.jpg")
                if os.path.exists(img_path):
                    image_paths.append(img_path)
                    found = True
                    break  # AutoMM может работать с несколькими изображениями, но для простоты возьмем первое
            if not found:
                image_paths.append(None)  # Если изображений нет, добавляем None
        df['image'] = image_paths
        return df

    # Добавляем пути к изображениям
    train_df = add_image_paths(train_df, train_img_dir)
    test_df = add_image_paths(test_df, test_img_dir)
    
    return train_df, test_df

# Загрузка данных
train_df = pd.read_parquet(TRAIN_DATA_PATH)
test_df = pd.read_parquet(TEST_DATA_PATH)

# Подготовка данных (добавление путей к изображениям)
train_data, test_data = prepare_data(train_df, test_df, TRAIN_IMG_DIR, TEST_IMG_DIR)

# Убедимся, что целевая переменная - price_TARGET
label_column = 'price_TARGET'

# Шаг 3: Создание и конфигурация Predictor
# Указываем проблему как 'regression', целевую переменную и метрику
predictor = MultiModalPredictor(
    label=label_column,
    problem_type="regression",
    eval_metric="mean_absolute_percentage_error",  # MAPE близка к medianAPE
    path="automm_avito_car_price"  # Папка для сохранения моделей
)

# Шаг 4: Обучение модели
# AutoMM автоматически определит типы данных (изображения, текст, категориальные и числовые признаки)
predictor.fit(
    train_data=train_data,
    hyperparameters={
        "optimization.max_epochs": 10,  # Можно увеличить для лучшего качества
        "optimization.learning_rate": 1e-4,
        "env.per_gpu_batch_size": 8,  # Подберите в зависимости от доступной GPU памяти
        "model.names": ["hf_text", "timm_image", "categorical_mlp", "numerical_mlp"]  # Явно указываем модели для каждого типа данных
    }
)

# Шаг 5: Предсказание
predictions = predictor.predict(test_data)

# Создание файла для отправки
submission = pd.DataFrame({
    'ID': test_data.index,
    'target': predictions
})
submission.to_csv('submission_autogluon.csv', index=False)

print("Сабмит готов!")
