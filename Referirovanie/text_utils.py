import re
from typing import List, Tuple
import pymorphy3

morph = pymorphy3.MorphAnalyzer()

WORD_PATTERN = re.compile(r"[А-Яа-яA-Za-zЁё0-9\-]+", re.UNICODE)


def split_sentences(text: str) -> List[str]:
    """
    Простое разбиение на предложения по . ! ?
    Сохраняем исходный текст предложений для финального реферата.
    """
    text = text.strip()
    if not text:
        return []

    # Разбиваем по знакам конца предложения + пробелам
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Убираем пустые
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def tokenize(text: str) -> List[str]:
    """
    Токенизация: достаем слова/числа/дефисные формы.
    """
    return WORD_PATTERN.findall(text.lower())


def lemmatize_word(word: str) -> str:
    """
    Лемматизация одного слова.
    """
    parsed = morph.parse(word)
    if not parsed:
        return word.lower()
    return parsed[0].normal_form


def tokenize_and_lemmatize(text: str) -> List[Tuple[str, str]]:
    """
    Возвращает список пар:
    (исходный_токен_в_нижнем_регистре, лемма)
    """
    tokens = tokenize(text)
    result = []
    for token in tokens:
        lemma = lemmatize_word(token)
        result.append((token, lemma))
    return result