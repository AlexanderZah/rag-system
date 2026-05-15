import re


def clean_text(text: str) -> str:
    text = re.sub(r'Ошибка! Источник ссылки не найден\.', '', text)
    text = re.sub(
        r'^\\s*Версия\\s*[\\d.]+\\s*Руководство пользователя\\s*Страница\\s*\\d+', '', text, flags=re.MULTILINE)
    return text.strip()


def remove_table_of_contents(text: str) -> str:
    match = re.search(r'СОДЕРЖАНИЕ', text, re.IGNORECASE)
    if not match:
        return text
    lines = text[match.end():].splitlines()
    for i, line in enumerate(lines):
        if re.search(r'^\d+\.\s+[А-Я]', line):
            return "\n".join(lines[i:])
    return text


def cut_to_main_content(text: str) -> str:
    text = remove_table_of_contents(text)
    match = re.search(r'\n1\s*\.\s+[А-ЯA-Z]', text)
    if match:
        return text[match.start():]
    return text


def is_gibberish(text: str, threshold: float = 0.2) -> bool:
    clean = re.sub(r'[\s\d]', '', text)
    if not clean:
        return True
    cyr = sum(1 for c in clean if 'а' <= c.lower() <= 'я' or c in 'ёЁ')
    return cyr / len(clean) < threshold
