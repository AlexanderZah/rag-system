from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import logging
from typing import List

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Ты — помощник для Финансовой системы РЦК. 
Отвечай **только** на русском языке, кратко, по делу и структурировано.
Используй **только** информацию из предоставленного контекста.
Для каждого факта/шага указывай источник в формате [1], [2] и т.д.
Если информации недостаточно — честно скажи, чего именно не хватает.
Не придумывай факты."""


class EnhancedGenerator:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        self.llm = self._get_llm(model_name)

    def _get_llm(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_4bit=True
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            temperature=0.0,
            do_sample=False,
            top_p=1.0,
        )
        return HuggingFacePipeline(pipeline=pipe)

    def generate(self, query: str, context_docs: List[Document]) -> str:
        context_str = "\n\n".join([
            f"Чанк {i+1} [Источник {i+1}]:\n{doc.page_content}"
            for i, doc in enumerate(context_docs)
        ])

        prompt = f"""{SYSTEM_PROMPT}

Контекст:
{context_str}

Вопрос: {query}

Ответь структурировано, ссылаясь на источники [1], [2] и т.д.:
"""
        response = self.llm.invoke(prompt)
        return response
