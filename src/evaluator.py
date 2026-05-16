import json
from typing import Dict, List, Any
from langchain_core.documents import Document
from langchain_huggingface import HuggingFacePipeline
import pandas as pd
from tqdm import tqdm
import json
import numpy as np
import re


class RAGEvaluator:
    def __init__(self, llm_judge: HuggingFacePipeline):
        self.llm = llm_judge
        self.prompt_template = self._create_evaluation_prompt()

    def _create_evaluation_prompt(self) -> str:
        return """Ты — эксперт по оценке качества вопросно-ответных систем на русском языке в домене банковской и финансовой документации.

Оцени ответ RAG-системы строго по четырём метрикам. Используй ТОЛЬКО предоставленные данные.

**Входные данные:**
Контекст (Чанк 1, Чанк 2, ...):
{context}

Вопрос:
{question}

Ответ системы:
{answer}

Эталонный ответ:
{ground_truth}

---

### 1. Faithfulness (Достоверность)
Выдели из **Ответа системы** все атомарные фактологические утверждения.
Для каждого проверь, подтверждается ли оно **явно** в Контексте.
score = (подтверждённые / всего утверждений)

### 2. Answer Relevancy (Релевантность ответа)
Насколько ответ соответствует сути вопроса?
- 1.0 — точно и полно отвечает, без лишнего
- 0.7-0.9 — в целом по делу, есть небольшие избыточности
- 0.4-0.6 — частично релевантен
- 0.0-0.3 — слабо или совсем не по делу

### 3. Context Precision (Точность контекста)
Сколько из предоставленных чанков действительно полезны для ответа на вопрос?

### 4. Context Recall (Полнота контекста)
Какая доля фактов из **эталонного ответа** покрывается Контекстом?

---

**Ответь ТОЛЬКО валидным JSON** без дополнительного текста:

```json
{
  "faithfulness": {
    "score": 0.85,
    "claims_total": 12,
    "claims_supported": 10,
    "reasoning": "Перечисление утверждений с поддержкой..."
  },
  "answer_relevancy": {
    "score": 0.92,
    "reasoning": "Краткое обоснование"
  },
  "context_precision": {
    "score": 0.80,
    "relevant_chunks": 4,
    "total_chunks": 5,
    "reasoning": "Анализ каждого чанка"
  },
  "context_recall": {
    "score": 0.88,
    "covered_claims": 7,
    "total_gt_claims": 8,
    "reasoning": "Анализ эталонного ответа"
  }
}
```
"""

    def evaluate(
        self,
        question: str,
        answer: str,
        context_docs: List[Document],
        context_str,
        ground_truth: str
    ) -> Dict[str, Any]:
        """Оценка одного примера"""
        if not context_str:
            # Формируем контекст
            context_str = "\n\n".join([
                f"Чанк {i+1}:\n{doc.page_content}"
                for i, doc in enumerate(context_docs)
            ])

        prompt = self.prompt_template.format(
            context=context_str,
            question=question,
            answer=answer,
            ground_truth=ground_truth
        )

        try:
            response = self.llm.invoke(prompt)

            # Извлекаем JSON из ответа (на случай, если модель добавила текст)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            json_str = json_str.replace('\n', ' ').replace('\r', ' ').strip()
            json_str = re.sub(r'\s+', ' ', json_str)

            result = json.loads(json_str)
            return result

        except Exception as e:
            print(f"Ошибка при оценке: {e}")
            return {
                "faithfulness": {"score": 0.0, "reasoning": "Ошибка парсинга"},
                "answer_relevancy": {"score": 0.0, "reasoning": "Ошибка парсинга"},
                "context_precision": {"score": 0.0, "reasoning": "Ошибка парсинга"},
                "context_recall": {"score": 0.0, "reasoning": "Ошибка парсинга"}
            }

    def evaluate_dataset(
        self,
        dataset_path: str,
        retriever,
        generator,
        output_path: str = "evaluation_results.json"
    ):
        """Оценка всего датасета"""

        df = pd.read_excel(dataset_path)
        results = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Оценка датасета"):
            question = row['question']
            ground_truth = row['answer']

            # Получаем контекст
            docs = retriever.retrieve(question, k=5)

            # Генерируем ответ
            context_text = "\n\n".join([d.page_content for d in docs])
            answer = generator.generate(question, context_text)

            # Оцениваем
            eval_result = self.evaluate(question, answer, docs, ground_truth)

            record = {
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
                "metrics": eval_result
            }
            results.append(record)

        # Сохраняем результаты
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # Вычисляем средние метрики
        self._print_average_metrics(results)
        return results

    def _print_average_metrics(self, results):

        faithfulness = [r['metrics']['faithfulness']['score'] for r in results]
        relevancy = [r['metrics']['answer_relevancy']['score']
                     for r in results]
        precision = [r['metrics']['context_precision']['score']
                     for r in results]
        recall = [r['metrics']['context_recall']['score'] for r in results]

        print("\n" + "="*60)
        print("СРЕДНИЕ МЕТРИКИ ПО ДАТАСЕТУ")
        print("="*60)
        print(f"Faithfulness     : {np.mean(faithfulness):.4f}")
        print(f"Answer Relevancy : {np.mean(relevancy):.4f}")
        print(f"Context Precision: {np.mean(precision):.4f}")
        print(f"Context Recall   : {np.mean(recall):.4f}")
        print(
            f"Q (интегральная) : {(np.mean(faithfulness) + np.mean(relevancy) + np.mean(precision) + np.mean(recall))/4:.4f}")
        print("="*60)
