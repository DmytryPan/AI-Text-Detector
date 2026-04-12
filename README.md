# AI-Text-Detector

> Инструменты и методики для детекции текстов, сгенерированных большими языковыми моделями

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)
![Sklearn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![RAPIDS](https://img.shields.io/badge/RAPIDS-GPU_Accelerated-76B900?logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-green)

---

## О проекте

**AI-Text-Detector** — набор различных экспериментов и методик для классификации текстов на предмет их происхождения: написан ли текст человеком или сгенерирован искусственным интеллектом.

Проект включает в себя:
- Разведочный анализ данных
- Обучение трансформерных моделей (BERT, DeBERTa, RoBERTa, phi-3)
- Логистическая регрессия на эмбеддингах (E5-эмбеддинги + LogisticRegression из RAPIDS)
- Градиентный бустинг на эмбеддингах (E5-эмбеддинги + CatBoost)
- Fine-tuning компактных LLM (Phi-3)
- Калибровку порогов и анализ метрик


В качестве датасета для обучения используется [LLMTrace](https://huggingface.co/datasets/iitolstykh/LLMTrace_classification)
---
## Структура репозитория

├── notebooks/

│ ├── llmtrace-eda.ipynb # Разведочный анализ данных

│ ├── llmtrace-bert-base-cased.ipynb # BERT-base fine-tuning

│ ├── llmtrace-distilbert-base-multilingual-cased.ipynb # Multilingual DistilBERT

│ ├── llmtrace-mdeberta-v3-base.ipynb # DeBERTa-v3 fine-tuning

│ ├── llmtrace-xlm-roberta.ipynb # XLM-RoBERTa для кросс-языковой детекции

│ ├── llmtrace-build-e5-embeddings.ipynb # Извлечение E5-эмбеддингов

│ ├── llmtrace-catboost-e5-embeddings.ipynb # CatBoost на эмбеддингах

│ ├── llmtrace-logistic-regression.ipynb # Логистическая регрессия (baseline)

│ ├── llmtrace-distilbert-integratedgradient.ipynb # Explainable AI с Captum

│ ├── distilbert-threshold-calibrating.ipynb # Калибровка порога классификации

│ ├── llmtrace-phi-3-5-classificator.ipynb # Fine-tuning Phi-3.5

│ └── llmtrace-phi-3-inference-metrics.ipynb # Инференс и метрики Phi-3.5

├── .gitignore

└── README.md
---
## Датасет

Для обучения и оценки моделей используется датасет **[LLMTrace](https://huggingface.co/datasets/iitolstykh/LLMTrace_classification)**.


| Параметр | Значение |
|----------|----------|
| **Объём (train)** | ~411 440 примеров |
| **Объём (validation)** | ~86 696 примеров |
| **Объём (test)** | ~90 950 примеров |
| **Языки** | 🇷🇺 Русский, 🇬🇧 Английский |
| **Классы** | `ai` (ИИ-генерация), `human` (человеческий текст) |
| **ИИ-модели** | 30+ различных моделей (GPT-4o, Llama, Gemma, YandexGPT и др.) |
| **Типы данных** | статьи, стихи, отзывы, вопросы, истории, факты |
| **Баланс классов (train)** | ai: 246 491, human: 164 449 (60% vs 40%)|


---

## Методологии

### Transformer-based Fine-tuning

**Модели:**
 * distilbert/distilbert-base-multilingual-cased
 * google-bert/bert-base-multilingual-cased
 * microsoft/mdeberta-v3-base
 * FacebookAI/xlm-roberta-base

Модели семейства BERT дообучаются для бинарной классификации:

```python
# Архитектура классификатора
BertForSequenceClassification(
  (bert): BertModel(...)
  (classifier): Linear(in_features=768, out_features=2)
)
```

### Embedding + Gradient Boosting
**Двухэтапный пайплайн:**
1. Извлечение эмбеддингов с помощью модели intfloat/multilingual-e5-large (768-dim векторы)
2. Обучение классификаторов LogisticRegression и CatBoostClassifier на полученных эмбеддингах

### LLM-based Classification 

С помощью LoRa адаптеров над phi-3-mini-instruct-4k и дополнительного линейного head'a для бинарной классификации LLM была дообучена для решения задачи бинарной классификации. В качестве [CLS] эмбеддинга использовался эмбеддинг последнего токена.

```# Архитектура классификатора на базе Phi-3-mini
Phi3ForSequenceClassification(
  (model): Phi3Model(...)
  (classification_head): Linear(3072 → 2)
)
```

---

## Результаты классификации

### Сводная таблица метрик

В таблице представлены метрики классификации для положительного класса (ai):

| Модель | Accuracy | Precision | Recall | F1 | 
|--------|----------|-----------|--------|----|
| distilbert-base-multilingual-cased |0.93|0.93|0.96|0.94|
| bert-base-multilingual-cased | 0.92 | 0.91 | 0.96 | 0.93 |
| mdeberta-v3-base | 0.92 | 0.91 | **0.97** | 0.94 | 
| xlm-roberta-base | 0.91 | 0.88 | **0.97** | 0.92 | 
| Logistic Regression | 0.84 | 0.86 | 0.87 | 0.86 |
| CatBoostClassifier | 0.87 | 0.89 | 0.89 | 0.89 | 
| phi-3-mini | **0.94** | **0.97** | 0.93 | **0.95** |

Лучше всех по F1 мере себя показал подход при использовании LLM в качестве экстрактора признаков. Стоит отметить, что для достижения такого качества классификации модели потребовалось лишь дообучиться на 40000 примерах из обучающего набора.

### Матрица ошибок (Phi-3.5, test set)

| Истинный класс \ Предсказано | ИИ (Predicted) | Человек (Predicted) | Всего (Support) |
| :--- | :---: | :---: | :---: |
| **ИИ (Actual)** | **50,778** | 3,706 | 54,484 |
| **Человек (Actual)** | 1,654 | **34,812** | 36,466 |
| **Итого** | 52,432 | 38,518 | **90,950** |

---

## Ключевые наблюдения
1. Phi-3.5 показывает наилучшие результаты благодаря способности улавливать семантические паттерны. Модель как правило ошибается в строгих текстах с "сухой" лексикой. Также ей непросто ловить тексты из домена story. ИИ хорошо пишет художественное повествование, которое сложно отличить от человека.
2. CatBoost + E5 — отличный компромисс между качеством и скоростью инференса
3. Мультиязычные модели эффективно работают с текстами на русском и английском

---

## Источники


```@article{Layer2025LLMTrace,
  Title = {{LLMTrace: A Corpus for Classification and Fine-Grained Localization of AI-Written Text}},
  Author = {Irina Tolstykh and Aleksandra Tsybina and Sergey Yakubson and Maksim Kuprashevich},
  Year = {2025},
  Eprint = {arXiv:2509.21269}
}```
