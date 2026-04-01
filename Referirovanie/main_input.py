from summarizer import summarize_text
from rouge_metrics import evaluate_rouge


def build_abstracts(texts):
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


def parse_input_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return [], []

    if "TEXTS:" not in content:
        raise ValueError("В файле должен быть блок TEXTS:")

    if "REFERENCES:" in content:
        texts_part, references_part = content.split("REFERENCES:", 1)
    else:
        texts_part = content
        references_part = None

    texts_part = texts_part.replace("TEXTS:", "", 1).strip()


    texts = [doc.strip() for doc in texts_part.split("===") if doc.strip()]

    references = []
    if references_part is not None:
        references_part = references_part.strip()
        references = [ref.strip() for ref in references_part.split("===") if ref.strip()]

    return texts, references


if __name__ == "__main__":
    input_file = "input.txt"

    texts, references = parse_input_file(input_file)

    if not texts:
        print("Во входном файле нет текстов для обработки.")
        exit()

    summaries = build_abstracts(texts)

    print("Рефераты:")
    print("[")
    for i, summary in enumerate(summaries):
        end_symbol = "," if i < len(summaries) - 1 else ""
        safe_summary = summary.replace('"', '\\"')
        print(f'  "{safe_summary}"{end_symbol}')
    print("]")

    if references:
        if len(references) != len(summaries):
            raise ValueError("Количество references должно совпадать с количеством texts")

        scores = evaluate_rouge(summaries, references)
        print_rouge_scores(scores)