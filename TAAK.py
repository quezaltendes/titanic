import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import ViTImageProcessor, ViTModel
import os
import joblib
from tqdm.auto import tqdm
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Константы
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
VIT_MODEL_NAME = 'google/vit-huge-patch14-224-in21k'
EMBEDDING_DIM = 1280  # Для ViT-Huge

# Загрузка данных
train_df = pd.read_parquet('train_dataset.parquet')
test_df = pd.read_parquet('test_dataset.parquet')
sample_sub = pd.read_csv('sample_submission.csv')

# Пути к изображениям
TRAIN_IMG_PATH = '/kaggle/input/cv-avito-car-price-prediction/train_images/train_images'
TEST_IMG_PATH = '/kaggle/input/cv-avito-car-price-prediction/test_images/test_images'
def extract_embeddings(dataloader, model, device):
    emb_dict = {}
    
    with torch.no_grad():
        for batch, car_ids in tqdm(dataloader, desc="Extracting embeddings"):
            # batch имеет размер [batch_size, num_images, channels, height, width]
            batch = batch.to(device)
            batch_size, num_imgs, C, H, W = batch.shape
            
            # Преобразуем в [batch_size * num_imgs, channels, height, width]
            batch = batch.view(batch_size * num_imgs, C, H, W)
            
            # Получаем эмбеддинги [batch_size * num_imgs, embedding_dim]
            outputs = model(batch)
            embeddings = outputs.last_hidden_state[:, 0, :]  # Берем [CLS] токен
            
            # Возвращаем исходную размерность [batch_size, num_imgs, embedding_dim]
            embeddings = embeddings.view(batch_size, num_imgs, -1)
            
            # Агрегируем по изображениям (усредняем)
            aggregated_emb = embeddings.mean(dim=1).cpu().numpy()
            
            for i, car_id in enumerate(car_ids):
                emb_dict[car_id] = aggregated_emb[i]
                
    return emb_dict

# Создаем датасеты и загрузчики
train_dataset = CarDataset(train_df, TRAIN_IMG_PATH, processor)
test_dataset = CarDataset(test_df, TEST_IMG_PATH, processor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Извлекаем эмбеддинги (это может занять много времени)
print("Извлечение эмбеддингов для тренировочной выборки...")
train_embeddings = extract_embeddings(train_loader, vit_model, DEVICE)

print("Извлечение эмбеддингов для тестовой выборки...")
test_embeddings = extract_embeddings(test_loader, vit_model, DEVICE)

# Сохраним эмбеддинги, чтобы не пересчитывать каждый раз
joblib.dump(train_embeddings, 'train_embeddings_vit_huge.pkl')
joblib.dump(test_embeddings, 'test_embeddings_vit_huge.pkl')
# Загрузка сохраненных эмбеддингов (если нужно)
# train_embeddings = joblib.load('train_embeddings_vit_huge.pkl')
# test_embeddings = joblib.load('test_embeddings_vit_huge.pkl')

# Создаем DataFrame из эмбеддингов
train_emb_df = pd.DataFrame.from_dict(train_embeddings, orient='index')
train_emb_df.columns = [f'vit_emb_{i}' for i in range(train_emb_df.shape[1])]
train_emb_df.index.name = 'id'

test_emb_df = pd.DataFrame.from_dict(test_embeddings, orient='index')
test_emb_df.columns = [f'vit_emb_{i}' for i in range(test_emb_df.shape[1])]
test_emb_df.index.name = 'id'

# Объединяем с основными данными
train_full = train_df.join(train_emb_df, how='left')
test_full = test_df.join(test_emb_df, how='left')

# Целевая переменная
y = train_full['price_TARGET'].copy()
X = train_full.drop('price_TARGET', axis=1)
X_test = test_full.copy()
# Выберем категориальные и числовые признаки (пример, нужно дополнить)
categorical_cols = ['body_type', 'drive_type', 'engine_type', 'color', 'pts', 'steering_wheel', 'equipment']
numerical_cols = ['doors_number', 'crashes_count', 'owners_count', 'mileage', 'latitude', 'longitude']

# Одновыборные опции (предполагаем, что это категориальные)
single_option_cols = ['audiosistema', 'diski', 'electropodemniki', 'fary', 'salon', 'upravlenie_klimatom', 'usilitel_rul']
categorical_cols += single_option_cols

# Мультивыборные поля нужно как-то обработать. Один из вариантов - создать бинарные признаки для самых частых опций.
# Для бейзлайна можно просто проигнорировать или создать признак "количество опций".
multiselect_cols = [col for col in X.columns if 'mult' in col]

# Для простоты добавим количество опций в каждом мультиселекте
for col in multiselect_cols:
    # Предполагаем, что данные хранятся как список или строка
    X[f'{col}_count'] = X[col].apply(lambda x: len(x) if isinstance(x, list) else (x.count(',') + 1 if x != '[None]' else 0))
    X_test[f'{col}_count'] = X_test[col].apply(lambda x: len(x) if isinstance(x, list) else (x.count(',') + 1 if x != '[None]' else 0))

numerical_cols += [f'{col}_count' for col in multiselect_cols]

# Кодируем категориальные признаки
label_encoders = {}
for col in categorical_cols:
    # Объединяем train и test для consistent encoding
    all_data = pd.concat([X[col], X_test[col]], axis=0)
    le = LabelEncoder()
    le.fit(all_data.astype(str))
    
    X[col] = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

# В итоговый набор признаков включаем все обработанные колонки + ViT эмбеддинги
feature_cols = categorical_cols + numerical_cols + list(train_emb_df.columns)

X_train = X[feature_cols]
X_test_final = X_test[feature_cols]

# Заполним пропуски (если есть)
X_train = X_train.fillna(X_train.median(numeric_only=True))
X_test_final = X_test_final.fillna(X_train.median(numeric_only=True))
# Логарифмируем целевую переменную
y_log = np.log1p(y)

# Разделим на тренировочную и валидационную выборки для оценки модели
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_log, test_size=0.2, random_state=42)

# Создаем датасеты для LightGBM
train_data = lgb.Dataset(X_tr, label=y_tr)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Параметры LightGBM
params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 128,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'seed': 42,
    'max_depth': -1,
}

# Обучаем модель
model = lgb.train(params,
                 train_data,
                 num_boost_round=1000,
                 valid_sets=[val_data],
                 callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)])

# Предсказание на тесте (не забываем преобразовать обратно из логарифма)
y_test_pred_log = model.predict(X_test_final)
y_test_pred = np.expm1(y_test_pred_log)
# Создаем DataFrame с предсказаниями
submission = pd.DataFrame({
    'ID': X_test_final.index,
    'target': y_test_pred
})

# Убедимся, что порядок ID совпадает с sample_submission
submission = submission.set_index('ID').reindex(sample_sub['ID']).reset_index()

# Сохраняем
submission.to_csv('submission_vit_huge_lgb.csv', index=False)
print("Сабмит сохранен!")
