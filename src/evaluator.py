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
    def __init__(self, llm_judge: HuggingFacePipeline, tokenizer):
        self.llm = llm_judge
        self.tokenizer = tokenizer
        self.prompt_template = self._create_evaluation_prompt()

    def evaluate(
        self,
        question: str,
        answer: str,
        context_docs: List[Document],
        context_str: str,
        ground_truth: str
    ) -> Dict[str, Any]:

        if not context_str:
            context_str = "\n\n".join([
                f"Чанк {i+1}:\n{doc.page_content}"
                for i, doc in enumerate(context_docs)
            ])

        system_prompt = (
            "Ты — эксперт по оценке качества вопросно-ответных систем на русском языке "
            "в домене банковской и финансовой документации.\n\n"
            "Оцени ответ RAG-системы строго по четырём метрикам. Используй ТОЛЬКО предоставленные данные.\n\n"
            "---\n\n"
            "### 1. Faithfulness (Достоверность)\n"
            "Выдели из **Ответа системы** все атомарные фактологические утверждения.\n"
            "Для каждого проверь, подтверждается ли оно **явно** в Контексте.\n"
            "score = (подтверждённые / всего утверждений)\n\n"
            "### 2. Answer Relevancy (Релевантность ответа)\n"
            "Насколько ответ соответствует сути вопроса?\n"
            "- 1.0 — точно и полно отвечает, без лишнего\n"
            "- 0.7-0.9 — в целом по делу, есть небольшие избыточности\n"
            "- 0.4-0.6 — частично релевантен\n"
            "- 0.0-0.3 — слабо или совсем не по делу\n\n"
            "### 3. Context Precision (Точность контекста)\n"
            "Сколько из предоставленных чанков действительно полезны для ответа на вопрос?\n\n"
            "### 4. Context Recall (Полнота контекста)\n"
            "Какая доля фактов из **эталонного ответа** покрывается Контекстом?\n\n"
            "---\n\n"
            "Верни **строго** JSON со следующей структурой:\n"
            "```json\n"
            "{\n"
            '  "faithfulness": {\n'
            '    "score": <float от 0 до 1>,\n'
            '    "claims_total": <int>,\n'
            '    "claims_supported": <int>,\n'
            '    "reasoning": "<строка>"\n'
            '  },\n'
            '  "answer_relevancy": {\n'
            '    "score": <float от 0 до 1>,\n'
            '    "reasoning": "<строка>"\n'
            '  },\n'
            '  "context_precision": {\n'
            '    "score": <float от 0 до 1>,\n'
            '    "relevant_chunks": <int>,\n'
            '    "total_chunks": <int>,\n'
            '    "reasoning": "<строка>"\n'
            '  },\n'
            '  "context_recall": {\n'
            '    "score": <float от 0 до 1>,\n'
            '    "covered_claims": <int>,\n'
            '    "total_gt_claims": <int>,\n'
            '    "reasoning": "<строка>"\n'
            '  }\n'
            '}\n'
            "```\n"
            "Вычисли все значения на основе анализа предоставленных данных. "
            "Не используй числа из примера – они даны только для демонстрации формата."
        )

        # 3. Пользовательский запрос – только данные
        user_prompt = f"""**Входные данные:**
    Контекст (Чанк 1, Чанк 2, ...):
    {context_str}

    Вопрос:
    {question}

    Ответ системы:
    {answer}

    Эталонный ответ:
    {ground_truth}"""

        # 4. Применяем чат-шаблон, специфичный для Qwen2.5-Instruct
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(formatted_prompt, return_tensors='pt')

        response = self.llm.invoke(formatted_prompt)

        match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Запасной вариант – ищем первую фигурную скобку
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                raise ValueError("JSON не найден в ответе модели")
            json_str = response[json_start:json_end]

        try:
            result = json.loads(json_str)
            return result
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Ошибка парсинга JSON: {e}\nСырой ответ модели:\n{response}")

    def evaluate_dataset(
        self,
        dataset_path: str,
        retriever,
        generator,
        output_path: str = "evaluation_results.json"
    ):

        df = pd.read_excel(dataset_path)
        results = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Оценка датасета"):
            question = row['question']
            ground_truth = row['answer']

            docs = retriever.retrieve(question, k=5)

            context_text = "\n\n".join([d.page_content for d in docs])
            answer = generator.generate(question, context_text)

            eval_result = self.evaluate(question, answer, docs, ground_truth)

            record = {
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
                "metrics": eval_result
            }
            results.append(record)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

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
