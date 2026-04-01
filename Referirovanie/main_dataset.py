from typing import List, Tuple

from datasets import load_dataset

from summarizer import summarize_text
from rouge_metrics import evaluate_rouge


def build_abstracts(texts: List[str]) -> List[str]:
    """
    Вход: массив текстов
    Выход: массив рефератов
    """
    return [summarize_text(text) for text in texts]


def print_rouge_scores(scores):
    """
    Явный вывод ROUGE-1 / ROUGE-2 / ROUGE-L
    """
    print("\nROUGE scores:")
    for metric_name in ["rouge-1", "rouge-2", "rouge-l"]:
        metric = scores[metric_name]
        print(
            f"{metric_name.upper()}: "
            f"Precision={metric['precision']:.4f}, "
            f"Recall={metric['recall']:.4f}, "
            f"F1={metric['f1']:.4f}"
        )


def load_gazeta_dataset(split: str = "test", limit: int = None) -> Tuple[List[str], List[str]]:
    """
    Загрузка датасета Gazeta с Hugging Face через datasets.

    split:
        "train", "validation", "test"

    limit:
        если задан, берем только первые limit записей
    """
    dataset = load_dataset("IlyaGusev/gazeta", split=split)

    texts = []
    references = []

    count = 0
    for item in dataset:
        text = item["text"].strip()
        summary = item["summary"].strip()

        if not text or not summary:
            continue

        texts.append(text)
        references.append(summary)

        count += 1
        if limit is not None and count >= limit:
            break

    return texts, references


def print_summaries_array(summaries: List[str]):
    print("\nРефераты:")
    print("[")
    for i, summary in enumerate(summaries):
        safe_summary = summary.replace("\\", "\\\\").replace('"', '\\"')
        end_symbol = "," if i < len(summaries) - 1 else ""
        print(f'  "{safe_summary}"{end_symbol}')
    print("]")


def print_examples(texts: List[str], references: List[str], summaries: List[str], count: int = 3):
    """
    Печатает несколько примеров для визуальной проверки.
    """
    n = min(count, len(texts), len(references), len(summaries))

    print("\nПримеры:")
    for i in range(n):
        print("=" * 80)
        print(f"Пример #{i + 1}")
        print(f"\nТекст (первые 500 символов):\n{texts[i][:500]}...")
        print(f"\nЭталонный summary:\n{references[i]}")
        print(f"\nСгенерированный summary:\n{summaries[i]}")
        print("=" * 80)


if __name__ == "__main__":

    dataset_split = "test"

    # Сколько примеров обработать:
    # None = весь сплит
    limit = 50

    # Сколько примеров показать
    examples_to_show = 5

    print(f"Загрузка датасета Gazeta (split='{dataset_split}')...")
    texts, references = load_gazeta_dataset(split=dataset_split, limit=limit)

    if not texts:
        raise ValueError("Не удалось загрузить данные из датасета Gazeta.")

    print(f"Загружено документов: {len(texts)}")


    print("Генерация рефератов...")
    summaries = build_abstracts(texts)


    print_summaries_array(summaries)


    print("\nВычисление ROUGE...")
    scores = evaluate_rouge(summaries, references)
    print_rouge_scores(scores)


    print_examples(texts, references, summaries, count=examples_to_show)