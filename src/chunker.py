from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
import re
from langchain_core.documents import Document


def split_by_structure(text: str):
    # Адаптивный чанкинг по заголовкам
    if re.search(r'(?m)^\d+\.\d+\.', text):
        pattern = r'(?ms)^(\d+(?:\.\d+)*\.\s+[А-ЯA-Z].*?)(?=^\d+(?:\.\d+)*\.\s+[А-ЯA-Z]|\Z)'
    else:
        pattern = r'(?ms)^(\d+\.\s+[А-ЯA-Z].*?)(?=^\d+\.\s+[А-ЯA-Z]|\Z)'

    matches = re.findall(pattern, text)
    if matches and all(len(m) > 100 for m in matches):
        return matches

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "]
    )
    return splitter.split_text(text)
