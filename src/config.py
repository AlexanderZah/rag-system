from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = ""

MODEL_EMBEDDINGS = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DB_INDEX_NAME = "sentence-transformers_paraphrase-multilingual-mpnet-base-v2"

NUMBER_RELEVANT_CHUNKS = 5
TOP_K_FOR_RERANK = 20
RERANKER_MODEL = "DiTy/cross-encoder-russian-msmarco"
TOP_K_CANDIDATES = 20
FINAL_K = 5
USE_RRF = True
