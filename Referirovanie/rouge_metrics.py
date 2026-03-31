# rouge_metrics.py

from collections import Counter
from typing import Dict, List, Tuple
import re


WORD_PATTERN = re.compile(r"[А-Яа-яA-Za-zЁё0-9\-]+", re.UNICODE)


def tokenize_for_rouge(text: str) -> List[str]:
    """
    Токенизация для ROUGE:
    - приводим к нижнему регистру
    - выделяем слова
    """
    return WORD_PATTERN.findall(text.lower())


def get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """
    Получить список n-грамм.
    """
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_n(summary: str, reference: str, n: int = 1) -> Dict[str, float]:
    """
    ROUGE-N:
    - считаем пересечение мультимножеств n-грамм
    - Precision = overlap / ngrams(summary)
    - Recall = overlap / ngrams(reference)
    - F1 = harmonic mean
    """
    summary_tokens = tokenize_for_rouge(summary)
    reference_tokens = tokenize_for_rouge(reference)

    summary_ngrams = get_ngrams(summary_tokens, n)
    reference_ngrams = get_ngrams(reference_tokens, n)

    summary_counter = Counter(summary_ngrams)
    reference_counter = Counter(reference_ngrams)

    overlap = 0
    for gram in summary_counter:
        overlap += min(summary_counter[gram], reference_counter.get(gram, 0))

    precision = safe_div(overlap, sum(summary_counter.values()))
    recall = safe_div(overlap, sum(reference_counter.values()))
    f1 = f1_score(precision, recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def lcs_length(a: List[str], b: List[str]) -> int:
    """
    Длина наибольшей общей подпоследовательности (LCS).
    Для ROUGE-L.
    """
    n = len(a)
    m = len(b)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[n][m]


def rouge_l(summary: str, reference: str) -> Dict[str, float]:
    """
    ROUGE-L:
    - based on LCS (longest common subsequence)
    """
    summary_tokens = tokenize_for_rouge(summary)
    reference_tokens = tokenize_for_rouge(reference)

    lcs = lcs_length(summary_tokens, reference_tokens)

    precision = safe_div(lcs, len(summary_tokens))
    recall = safe_div(lcs, len(reference_tokens))
    f1 = f1_score(precision, recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def evaluate_rouge(summaries: List[str], references: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Усредненная оценка по массиву документов:
    - ROUGE-1
    - ROUGE-2
    - ROUGE-L

    Важно: references должны быть "золотым стандартом" (ручные рефераты).
    """
    if len(summaries) != len(references):
        raise ValueError("Количество summary и reference должно совпадать")

    metrics = {
        "rouge-1": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "rouge-2": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "rouge-l": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
    }

    count = len(summaries)
    if count == 0:
        return metrics

    for summary, reference in zip(summaries, references):
        r1 = rouge_n(summary, reference, n=1)
        r2 = rouge_n(summary, reference, n=2)
        rl = rouge_l(summary, reference)

        for key in ("precision", "recall", "f1"):
            metrics["rouge-1"][key] += r1[key]
            metrics["rouge-2"][key] += r2[key]
            metrics["rouge-l"][key] += rl[key]

    for metric_name in metrics:
        for key in metrics[metric_name]:
            metrics[metric_name][key] /= count

    return metrics