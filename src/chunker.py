from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
import re
from langchain_core.documents import Document


def split_by_structure(text: str, max_chunk_size: int = 1200, chunk_overlap: int = 150) -> list[str]:

    heading_pattern = (
        r'(?:(?<=\n)|(?<=^))'                     # начало строки или текста
        r'(?:'
        r'\s*(?:\d+(?:\.\d+)+|[A-ZА-Я])\.\s+'  # нумерация: 1., 1.1., A., Б.
        r'|'
        r'(?:Глава|Раздел|Тема|Часть|Пункт)\s+'  # ключевые слова
        r'|'
        r'(?:[А-Я]{2,}\s*:\s)'                # ЗАГОЛОВОК:
        r')'
    )

    parts = re.split(rf'(?={heading_pattern})', text, flags=re.MULTILINE)

    if parts and parts[0].strip() == '':
        parts = parts[1:]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )

    final_chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) <= max_chunk_size:
            final_chunks.append(part)
        else:
            final_chunks.extend(splitter.split_text(part))

    return final_chunks
