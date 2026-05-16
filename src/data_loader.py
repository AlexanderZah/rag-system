import os
import sqlite3
import io
import re
from typing import List

from langchain_core.documents import Document as LCDocument
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import DocumentStream

from .utils import clean_text, remove_table_of_contents, cut_to_main_content, is_gibberish
from .chunker import split_by_structure


class RCKDataLoader:
    def __init__(self, embeddings: HuggingFaceEmbeddings, db_index_name: str):
        self.embeddings = embeddings
        self.db_index_name = db_index_name
        self.converter = self._init_docling_converter()

    def _init_docling_converter(self):
        pipeline_options = PdfPipelineOptions(
            do_ocr=False,
            do_table_structure=False,
            do_heading_detection=True
        )
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options)
            }
        )

    def load_pdf_with_docling(self, content: bytes, filename: str):
        """Загрузка PDF через Docling"""
        doc_stream = DocumentStream(
            name=filename,
            stream=io.BytesIO(content)
        )
        result = self.converter.convert(doc_stream)
        return result.document

    def process_document(self, row) -> List[LCDocument]:
        """Полная обработка одного PDF документа"""
        filename = row['filename']
        content = row['content']

        if not content:
            print(f'Пропуск пустого файла: {filename}')
            return []

        print(f'Обработка: {filename}')

        try:
            # 1. Docling
            docling_doc = self.load_pdf_with_docling(content, filename)
            full_text = docling_doc.export_to_text()

            if not full_text or is_gibberish(full_text[:1000]):
                print(f'Мусорный текст: {filename}')
                return []

            # 2. Очистка
            cleaned = clean_text(full_text)
            cleaned = remove_table_of_contents(cleaned)
            cleaned = cut_to_main_content(cleaned)

            # 3. Разбиение на чанки
            sections = split_by_structure(cleaned)

            if not sections:
                print(f'  Fallback splitter для {filename}')

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1200, chunk_overlap=150
                )
                sections = splitter.split_text(cleaned)

            print(f'  → Получено чанков: {len(sections)}')

            # 4. Создание LangChain документов
            docs = []
            for i, section in enumerate(sections):
                if len(section.strip()) < 50:
                    continue
                docs.append(
                    LCDocument(
                        page_content=section.strip(),
                        metadata={
                            "source": filename,
                            "db_id": row['id'],
                            "chunk_index": i,
                            "chunk_type": "instruction",
                            "created_at": row['created_at'],
                            "updated_at": row['updated_at'],
                            "size_bytes": row['size_bytes']
                        }
                    )
                )
            return docs

        except Exception as e:
            print(f'Ошибка обработки {filename}: {e}')
            return []

    def get_index_db(
        self,
        sqlite_path: str,
        force_rebuild: bool = False
    ) -> FAISS:
        """Основная функция создания/загрузки векторной базы"""

        if not force_rebuild and os.path.exists(self.db_index_name):
            print(f'Загрузка существующей базы: {self.db_index_name}')
            return FAISS.load_local(
                self.db_index_name,
                self.embeddings,
                allow_dangerous_deserialization=True
            )

        print('Создание новой векторной базы...')

        conn = sqlite3.connect(sqlite_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, filename, content, created_at, updated_at, size_bytes
            FROM instruction_files 
            WHERE file_type = 'pdf' OR filename LIKE '%.pdf'
        """)
        rows = cursor.fetchall()
        conn.close()

        print(f'Найдено PDF файлов: {len(rows)}')

        all_docs = []
        gibberish_count = 0

        for idx, row in enumerate(rows):
            print(f'[{idx+1}/{len(rows)}] {row["filename"]}')
            docs = self.process_document(row)
            all_docs.extend(docs)

        if not all_docs:
            raise ValueError("Не удалось извлечь ни одного чанка!")

        print(f'Всего чанков: {len(all_docs)}')
        print(f'Мусорных файлов: {gibberish_count}')

        # Создание FAISS
        print("Создание FAISS индекса...")
        db = FAISS.from_documents(all_docs, self.embeddings)

        # Сохранение
        os.makedirs(self.db_index_name, exist_ok=True)
        db.save_local(self.db_index_name)
        print(f'База успешно сохранена: {self.db_index_name}')

        return db
