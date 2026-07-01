# NIR Sentence Embeddings

Тема НИР: «Оценка способности русскоязычных sentence embeddings различать семантическую близость и логическое противоречие в парах предложений».

## Цель исследования

«Оценить, насколько cosine similarity между русскоязычными sentence embeddings позволяет различать семантически близкие, нейтральные и логически противоречивые пары предложений».

## Объект исследования

«Методы построения векторных представлений русскоязычных предложений в задачах семантического анализа текста».

## Предмет исследования

«Способность sentence embeddings отражать различия между отношениями семантического следования, нейтральности и противоречия в парах предложений».

## Исследовательский вопрос

Может ли простая мера cosine similarity между sentence embeddings надежно отличать пары с отношением entailment от contradiction, особенно когда contradiction-пары имеют высокую лексическую похожесть?

## Краткое описание эксперимента

В проекте используются русскоязычные пары предложений из XNLI Russian с разметкой `entailment`, `neutral`, `contradiction`.

Для каждой пары:

1. Загружается premise и hypothesis.
2. Считаются простые текстовые признаки: длина предложений и lexical overlap.
3. Строятся векторные представления с помощью TF-IDF и sentence-transformer модели.
4. Для каждой пары считается cosine similarity.
5. Similarity сравнивается между классами NLI.
6. Отдельно анализируются contradiction-пары с высокой similarity как потенциальные ошибки embedding-подхода.

## Структура репозитория

```text
nir-sentence-embeddings/
  README.md
  requirements.txt
  .gitignore
  .env.example
  notebooks/
    01_experiments.ipynb
  src/
    __init__.py
    data.py
    models.py
    evaluation.py
    visualization.py
  scripts/
    run_experiment.py
  outputs/
    .gitkeep
  reports/
    .gitkeep
```

## Локальный запуск

Создайте виртуальное окружение и установите зависимости:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Запустите эксперимент:

```bash
python scripts/run_experiment.py --n-per-class 300 --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Для быстрого теста можно уменьшить выборку:

```bash
python scripts/run_experiment.py --n-per-class 10
```

Полный запуск с нейросетевой моделью может занять время и потребовать загрузки модели с Hugging Face.

## Запуск в Google Colab

В начале notebook выполните:

```python
!git clone https://github.com/yaroslav2308/nir-sentence-embeddings.git
%cd nir-sentence-embeddings
!grep -v "^jupyter$" requirements.txt > /tmp/requirements-colab.txt
!pip install -r /tmp/requirements-colab.txt
```

В Colab пакет `jupyter` устанавливать не нужно: он может обновить `jupyter-server` и создать конфликт с пакетом `google-colab`.

После этого можно открыть и выполнить `notebooks/01_experiments.ipynb`.

## Планируемые модели

- `TF-IDF` как простой лексический baseline.
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` — быстрая мультиязычная модель для первого запуска.
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` — более тяжелая мультиязычная модель.
- `intfloat/multilingual-e5-base` — дополнительная модель, если понадобится.
- `ai-forever/ru-en-RoSBERTa` — русскоязычная embedding-модель, если она корректно запускается через `sentence-transformers` или `transformers`.

Не обязательно запускать все модели сразу. Для стартового эксперимента достаточно TF-IDF и одной SentenceTransformer-модели.

## Планируемые метрики

- Среднее значение cosine similarity по каждому классу.
- Медиана, стандартное отклонение, минимум и максимум similarity по каждому классу.
- Delta между средним similarity для `entailment` и `contradiction`.
- ROC-AUC для бинарного различения `entailment` против `contradiction`.
- Ручной анализ contradiction-пар с высокой cosine similarity.

## Ожидаемые артефакты

После запуска эксперимента в `outputs/` должны появиться:

- `results.csv` — таблица с исходными парами, признаками и similarity.
- `similarity_by_label.png` — график распределения cosine similarity по классам.
- `high_similarity_contradictions.csv` — примеры contradiction-пар с высокой cosine similarity.

Дополнительные выводы и интерпретацию результатов можно сохранять в `reports/`.
