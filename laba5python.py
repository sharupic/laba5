import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# ================= ЗАДАНИЕ 1: Загрузка и разбиение =================
print("1. Загрузка данных...")
df = pd.read_csv('creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']

# Разбиение с сохранением пропорции классов (stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Словарь для хранения моделей и их результатов
trained_models = {}
f1_scores = {}

print("\n--- Задание 1: Базовые модели ---")
base_models = {
    'Bagging': BaggingClassifier(estimator=DecisionTreeClassifier(class_weight='balanced'), n_estimators=30, random_state=42, n_jobs=-1),
    'RandomForest': RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42, n_jobs=-1),
    'LightGBM': LGBMClassifier(n_estimators=100, is_unbalance=True, random_state=42, verbose=-1, n_jobs=-1)
}

for name, model in base_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    f1_scores[name] = f1
    trained_models[name] = model
    print(f"{name}: macro F1 = {f1:.4f}")

# ================= ЗАДАНИЕ 2: Оптимизация LightGBM =================
print("\n--- Задание 2: Оптимизация (RandomizedSearchCV) ---")
lgbm_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [-1, 10]
}
lgbm_search = RandomizedSearchCV(
    LGBMClassifier(is_unbalance=True, random_state=42, verbose=-1, n_jobs=-1),
    lgbm_params, n_iter=4, cv=3, scoring='f1_macro', random_state=42, n_jobs=-1
)
lgbm_search.fit(X_train, y_train)
best_lgbm = lgbm_search.best_estimator_

# Оцениваем оптимизированный LightGBM
y_pred_opt = best_lgbm.predict(X_test)
f1_opt = f1_score(y_test, y_pred_opt, average='macro')
print(f"Оптимизированный LightGBM: macro F1 = {f1_opt:.4f}")

# Сохраняем оптимизированную модель, если она лучше базовой
if f1_opt > f1_scores['LightGBM']:
    f1_scores['LightGBM (opt)'] = f1_opt
    trained_models['LightGBM (opt)'] = best_lgbm

# ================= ЗАДАНИЕ 3: CatBoost =================
print("\n--- Задание 3: CatBoost ---")
cat_model = CatBoostClassifier(
    iterations=300, depth=6, learning_rate=0.1, 
    auto_class_weights='Balanced', verbose=0, random_state=42
)
cat_model.fit(X_train, y_train)
y_pred_cat = cat_model.predict(X_test)
f1_cat = f1_score(y_test, y_pred_cat, average='macro')
f1_scores['CatBoost'] = f1_cat
trained_models['CatBoost'] = cat_model
print(f"CatBoost: macro F1 = {f1_cat:.4f}")

# ================= ПОИСК ЛУЧШЕЙ МОДЕЛИ И МАТРИЦА ОШИБОК =================
print("\n" + "="*40)
best_model_name = max(f1_scores, key=f1_scores.get)
best_model = trained_models[best_model_name]
print(f"ЛУЧШАЯ МОДЕЛЬ: {best_model_name} (macro F1 = {f1_scores[best_model_name]:.4f})")
print("="*40)

# Строим матрицу ошибок для ЛУЧШЕЙ модели
print(f"\nСтроим матрицу ошибок для {best_model_name}...")
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Легальная (0)', 'Мошенническая (1)'],
            yticklabels=['Легальная (0)', 'Мошенническая (1)'])
plt.title(f'Матрица ошибок ({best_model_name})')
plt.xlabel('Предсказано')
plt.ylabel('Истина')
plt.tight_layout()
plt.savefig('best_confusion_matrix.png') # Сохранит картинку в папку
plt.show() # Покажет график на экране

print("\nВсе задания выполнены! Картинка матрицы ошибок сохранена в файл 'best_confusion_matrix.png'")