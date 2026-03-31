# summarizer.py

import re
from collections import Counter
from typing import List, Tuple, Set

from text_utils import split_sentences, tokenize_and_lemmatize
import config


# Токенизация с позициями в исходной строке (для сохранения пунктуации)
TOKEN_WITH_SPAN_PATTERN = re.compile(r"[А-Яа-яA-Za-zЁё0-9\-]+", re.UNICODE)


def _tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """
    Возвращает список токенов с позициями:
    (token_lower, start_char, end_char)
    """
    result = []
    for match in TOKEN_WITH_SPAN_PATTERN.finditer(text):
        token = match.group(0).lower()
        result.append((token, match.start(), match.end()))
    return result


def _get_significant_words(text: str) -> Set[str]:
    """
    Определяем значимые слова:
    - лемматизируем
    - считаем частоты
    - отбрасываем слишком редкие
    - берем верхний процент по частоте
    """
    token_lemma_pairs = tokenize_and_lemmatize(text)
    lemmas = [lemma for _, lemma in token_lemma_pairs]

    if not lemmas:
        return set()

    freq = Counter(lemmas)

    # Убираем слишком редкие
    filtered = {lemma: cnt for lemma, cnt in freq.items() if cnt >= config.MIN_WORD_FREQ}
    if not filtered:
        return set()

    sorted_items = sorted(filtered.items(), key=lambda x: x[1], reverse=True)

    top_count = max(1, int(len(sorted_items) * config.SIGNIFICANT_TOP_PERCENT))
    top_words = {lemma for lemma, _ in sorted_items[:top_count]}

    return top_words


def _build_sentence_data(sentence: str) -> List[Tuple[str, str, int, int]]:
    """
    Для предложения возвращает список:
    (token_lower, lemma, start_char, end_char)

    token_lower — токен в нижнем регистре
    lemma      — лемма токена
    start_char / end_char — позиции токена в исходном предложении
    """
    token_spans = _tokenize_with_spans(sentence)
    token_lemma_pairs = tokenize_and_lemmatize(sentence)

    # На всякий случай страхуемся от несовпадения длины
    n = min(len(token_spans), len(token_lemma_pairs))

    result = []
    for i in range(n):
        token_lower_from_span, start, end = token_spans[i]
        token_lower_from_lemma, lemma = token_lemma_pairs[i]

        # Обычно они совпадают, но берем токен из span-версии как источник позиций
        result.append((token_lower_from_span, lemma, start, end))

    return result


def _find_best_luhn_segment(sentence_data: List[Tuple[str, str, int, int]], significant_words: Set[str]) -> Tuple[float, int, int]:
    """
    Ищет лучший промежуток внутри предложения по методу Луна.

    Возвращает:
    (score, start_token_idx, end_token_idx)

    score = (число значимых слов)^2 / длина промежутка_в_токенах
    """
    significant_positions = [
        i for i, (_, lemma, _, _) in enumerate(sentence_data)
        if lemma in significant_words
    ]

    if not significant_positions:
        return 0.0, -1, -1

    best_score = 0.0
    best_start = -1
    best_end = -1

    n = len(significant_positions)

    for i in range(n):
        cluster = [significant_positions[i]]

        for j in range(i + 1, n):
            gap = significant_positions[j] - significant_positions[j - 1] - 1
            if gap <= config.MAX_NON_SIGNIFICANT_GAP:
                cluster.append(significant_positions[j])
            else:
                break

        start_idx = cluster[0]
        end_idx = cluster[-1]

        significant_count = len(cluster)
        span_len = end_idx - start_idx + 1

        if span_len <= 0:
            continue

        score = (significant_count ** 2) / span_len

        if score > best_score:
            best_score = score
            best_start = start_idx
            best_end = end_idx

    return best_score, best_start, best_end


def _expand_token_window(sentence_data: List[Tuple[str, str, int, int]], start_idx: int, end_idx: int, extra_tokens: int = 8) -> Tuple[int, int]:
    """
    Немного расширяет окно по токенам для читаемости.
    Всегда включает первые два токена предложения.
    """
    if start_idx < 0 or end_idx < 0:
        return start_idx, end_idx

    new_start = max(0, start_idx - extra_tokens)
    new_end = min(len(sentence_data) - 1, end_idx + extra_tokens)

    # Всегда включаем первые два токена предложения
    new_start = min(new_start, 2)

    return new_start, new_end


def _trim_fragment_boundaries(fragment: str) -> str:
    """
    Подчищает края фрагмента:
    - убирает лишние пробелы
    - убирает висящие запятые/тире/двоеточия/точки с запятой по краям
    """
    fragment = fragment.strip()

    # Убираем мусор на краях
    fragment = re.sub(r'^[\s,;:—\-–]+', '', fragment)
    fragment = re.sub(r'[\s,;:—\-–]+$', '', fragment)

    # Если осталась только пунктуация
    if not re.search(r'[А-Яа-яA-Za-zЁё0-9]', fragment):
        return ""

    return fragment.strip()


def _extract_fragment_from_original(sentence: str, sentence_data: List[Tuple[str, str, int, int]], start_idx: int, end_idx: int) -> str:
    """
    Вырезает фрагмент прямо из исходного предложения, сохраняя пунктуацию.
    """
    if start_idx < 0 or end_idx < 0 or not sentence_data:
        return ""

    _, _, start_char, _ = sentence_data[start_idx]
    _, _, _, end_char = sentence_data[end_idx]

    # Берем фрагмент из оригинального предложения
    fragment = sentence[start_char:end_char]

    # Чуть расширяем по символам до ближайших "естественных" границ,
    # но не захватываем слишком много.
    left = start_char
    right = end_char

    # Ищем слева ближайшую удобную границу (запятая, тире, начало)
    max_left_shift = 40
    i = start_char - 1
    steps = 0
    while i >= 0 and steps < max_left_shift:
        ch = sentence[i]
        if ch in ",;:—–-(":
            left = i + 1
            break
        i -= 1
        steps += 1
    else:
        # Если не нашли — оставляем по токену
        left = start_char

    # Ищем справа ближайшую удобную границу (запятая, тире, конец предложения)
    max_right_shift = 60
    i = end_char
    steps = 0
    while i < len(sentence) and steps < max_right_shift:
        ch = sentence[i]
        if ch in ",;:—–-)":
            right = i
            break
        if ch in ".!?":
            right = i
            break
        i += 1
        steps += 1
    else:
        right = end_char

    fragment = sentence[left:right]
    fragment = _trim_fragment_boundaries(fragment)

    return fragment


def _extract_best_fragment(sentence: str, significant_words: Set[str]) -> Tuple[float, str]:
    """
    Извлекает лучший фрагмент внутри предложения:
    - ищет лучший сегмент по Луну
    - расширяет окно по токенам
    - вырезает фрагмент ИЗ ОРИГИНАЛА (с пунктуацией)
    """
    sentence_data = _build_sentence_data(sentence)

    if not sentence_data:
        return 0.0, ""

    score, start_idx, end_idx = _find_best_luhn_segment(sentence_data, significant_words)

    if score < config.SENTENCE_SCORE_THRESHOLD or start_idx < 0 or end_idx < 0:
        return 0.0, ""

    # Чуть расширяем по токенам
    start_idx, end_idx = _expand_token_window(sentence_data, start_idx, end_idx, extra_tokens=12)

    fragment = _extract_fragment_from_original(sentence, sentence_data, start_idx, end_idx)

    if not fragment:
        return 0.0, ""

    return score, fragment


def _fragment_lemmas(fragment: str) -> Set[str]:
    """
    Множество лемм фрагмента для анти-дублирования.
    """
    return {lemma for _, lemma in tokenize_and_lemmatize(fragment)}


def _jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    """
    Jaccard similarity для анти-дублирования.
    """
    if not a or not b:
        return 0.0

    inter = len(a & b)
    union = len(a | b)

    if union == 0:
        return 0.0

    return inter / union


def summarize_text(text: str) -> str:
    """
    Основная функция реферирования по методу Луна:
    1. Разбиваем на предложения
    2. Находим значимые слова
    3. Для каждого предложения ищем лучший фрагмент
    4. Сортируем по score
    5. Добавляем анти-дублирование
    6. Ограничиваем длину итогового реферата
    """
    text = text.strip()
    if not text:
        return ""

    sentences = split_sentences(text)
    if not sentences:
        return ""

    significant_words = _get_significant_words(text)

    # Если значимых слов не нашли — fallback
    if not significant_words:
        fallback = sentences[0].strip()
        return fallback[:config.MAX_SUMMARY_LENGTH].strip()

    candidates = []  # (sentence_index, score, fragment)

    for idx, sentence in enumerate(sentences):
        score, fragment = _extract_best_fragment(sentence, significant_words)
        if score > 0 and fragment:
            candidates.append((idx, score, fragment))

    # Если ничего не прошло порог — fallback
    if not candidates:
        fallback = sentences[0].strip()
        return fallback[:config.MAX_SUMMARY_LENGTH].strip()

    # Сортируем по score по убыванию
    candidates.sort(key=lambda x: x[1], reverse=True)

    selected = []
    selected_lemma_sets = []
    current_length = 0

    # Порог анти-дублирования
    dedup_threshold = 0.6

    for idx, score, fragment in candidates:
        frag_lemmas = _fragment_lemmas(fragment)

        # Анти-дублирование
        is_duplicate = False
        for existing_lemmas in selected_lemma_sets:
            sim = _jaccard_similarity(frag_lemmas, existing_lemmas)
            if sim >= dedup_threshold:
                is_duplicate = True
                break

        if is_duplicate:
            continue

        # Ограничение длины итогового summary
        extra_len = len(fragment)
        if selected:
            extra_len += 1  # пробел между фрагментами

        if current_length + extra_len > config.MAX_SUMMARY_LENGTH:
            # Если еще ничего не выбрали — хотя бы обрежем лучший фрагмент
            if not selected:
                return fragment[:config.MAX_SUMMARY_LENGTH].strip()
            break

        selected.append((idx, fragment))
        selected_lemma_sets.append(frag_lemmas)
        current_length += extra_len

    # Если всё отфильтровалось
    if not selected:
        best_fragment = candidates[0][2]
        return best_fragment[:config.MAX_SUMMARY_LENGTH].strip()

    # Возвращаем в порядке появления в тексте
    selected.sort(key=lambda x: x[0])

    summary = " ".join(fragment for _, fragment in selected)

    # Финальная страховка по длине
    summary = summary[:config.MAX_SUMMARY_LENGTH].strip()

    # Возвращаем в порядке появления в тексте
    selected.sort(key=lambda x: x[0])

    # Склеиваем фрагменты через ". " для читаемости
    summary = ". ".join(fragment for _, fragment in selected)

    # Финальная страховка по длине
    summary = summary[:config.MAX_SUMMARY_LENGTH].strip()

    return summary