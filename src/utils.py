import re


def clean_text(text: str) -> str:
    text = re.sub(r'Ошибка! Источник ссылки не найден\.', '', text)
    text = re.sub(
        r'\|?\s*Страница\s+\d+.*?Дата создания\s+\d{2}\.\d{2}\.\d{4}\s*\|?', '', text)
    text = re.sub(r'^(ИС «РЦК»\s*(Валютный контроль|Управление расчетами|…)?\s*)$',
                  '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def remove_table_of_contents(text: str) -> str:
    """
    Удаляет блок СОДЕРЖАНИЕ до начала реального текста
    """

    start = re.search(r'СОДЕРЖАНИЕ', text)
    if not start:
        return text

    toc_line = r'.{0,100}\.{3,}\s*\d+\s*$'

    lines = text[start.end():].splitlines()
    end_idx = 0
    for i, line in enumerate(lines):
        if not re.search(toc_line, line):
            end_idx = i
            break

    return "\n".join(lines[end_idx:])


def cut_to_main_content(text: str) -> str:
    patterns = [
        r'(?m)^\s*(?:Глава|Раздел|Тема|Часть)\s+\d+\s*[\.:]?\s*\S',
        r'(?m)^\s*\d+(?:\.\d+)*\s+[А-ЯA-Z]',
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            return text[match.start():]
    return text


def is_gibberish(text: str, threshold: float = 0.2) -> bool:
    clean = re.sub(r'[\s\d]', '', text)
    if not clean:
        return True
    cyr = sum(1 for c in clean if 'а' <= c.lower() <= 'я' or c in 'ёЁ')
    return cyr / len(clean) < threshold
