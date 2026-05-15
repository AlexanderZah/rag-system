from src.config import *
from src.data_loader import RCKDataLoader
from src.retriever import AdvancedHybridRetriever, Reranker
from src.generator import EnhancedGenerator
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_EMBEDDINGS,
    model_kwargs={'device': 'cuda'}
)

loader = RCKDataLoader(embeddings=embeddings, db_index_name=DB_INDEX_NAME)
db = loader.get_index_db_from_sqlite(DB_PATH, force_rebuild=True)

# Получаем все документы
all_docs = list(db.docstore._dict.values())

retriever = AdvancedHybridRetriever(db, all_docs)
generator = EnhancedGenerator()

# Пример запроса
query = "Как заблокировать платёж перед отправкой в банк?"
docs = retriever.retrieve(query, k=5)
answer = generator.generate(query, docs)

print("=== Ответ ===")
print(answer)
print("\n=== Источники ===")
for i, d in enumerate(docs, 1):
    print(f"[{i}] {d.metadata.get('source', 'unknown')}")
