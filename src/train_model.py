from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from langchain_huggingface import HuggingFaceEmbeddings
import json

TRAIN_MODEL_PATH = "./finetuned_embedder"
TRAIN_DATA_PATH = "../data/train_embedder.json"

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")


with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
    train_data = json.load(f)


train_examples = []
for item in train_data:
    train_examples.append(InputExample(
        texts=[item["anchor"], item["positive"]]))


train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)


train_loss = losses.MultipleNegativesRankingLoss(model)


model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path=TRAIN_MODEL_PATH,
    show_progress_bar=True
)


finetuned_embeddings = HuggingFaceEmbeddings(
    model_name=TRAIN_MODEL_PATH,
    model_kwargs={'device': 'cuda'}
)
