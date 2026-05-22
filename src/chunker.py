from langchain_text_splitters import RecursiveCharacterTextSplitter
import re


def split_by_structure(text: str, max_chunk_size: int = 150, chunk_overlap: int = 50) -> list[str]:
    heading_pattern = (
        r'(?:(?<=\n)|(?<=^))'
        r'(?:'
        r'\s*(?:\d+(?:\.\d+)+|[A-ZА-Я])\.\s+'
        r'|'
        r'(?:Глава|Раздел|Тема|Часть|Пункт)\s+'
        r'|'
        r'(?:[А-Я]{2,}\s*:\s)'
        r')'
    )
    parts = re.split(rf'(?={heading_pattern})', text, flags=re.MULTILINE)
    parts = [p.strip() for p in parts if p.strip()]

    merged = []
    buffer = ''
    for part in parts:
        if len(buffer) + len(part) <= max_chunk_size:
            buffer += '\n' + part if buffer else part
        else:
            if buffer:
                merged.append(buffer)
            buffer = part
    if buffer:
        merged.append(buffer)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
    final_chunks = []
    for block in merged:
        if len(block) <= max_chunk_size:
            final_chunks.append(block)
        else:
            final_chunks.extend(splitter.split_text(block))
    return final_chunks
