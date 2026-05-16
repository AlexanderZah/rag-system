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

    def _escape_braces(self, text: str) -> str:
        """Экранирует фигурные скобки для безопасного форматирования строки."""
        return text.replace('{', '{{').replace('}', '}}')

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
    {{
    "faithfulness": {{
        "score": 0.85,
        "claims_total": 12,
        "claims_supported": 10,
        "reasoning": "Перечисление утверждений с поддержкой..."
    }},
    "answer_relevancy": {{
        "score": 0.92,
        "reasoning": "Краткое обоснование"
    }},
    "context_precision": {{
        "score": 0.80,
        "relevant_chunks": 4,
        "total_chunks": 5,
        "reasoning": "Анализ каждого чанка"
    }},
    "context_recall": {{
        "score": 0.88,
        "covered_claims": 7,
        "total_gt_claims": 8,
        "reasoning": "Анализ эталонного ответа"
    }}
    }}
    ```
    """

    def evaluate(
        self,
        question: str,
        answer: str,
        context_docs: List[Document],
        context_str: str,          # может быть None
        ground_truth: str
    ) -> Dict[str, Any]:
        """Оценка одного примера с использованием чат-шаблона Qwen2.5"""

        # 1. Формируем контекст, если не передан явно
        if not context_str:
            context_str = "\n\n".join([
                f"Чанк {i+1}:\n{doc.page_content}"
                for i, doc in enumerate(context_docs)
            ])

        # 2. Системная инструкция – полное описание метрик и формата JSON
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
            "**Ответь ТОЛЬКО валидным JSON** без дополнительного текста:\n\n"
            "```json\n"
            '{{\n'
            '"faithfulness": {{ \n'
            '    "score": 0.85, \n'
            '    "claims_total": 12, \n'
            '    "claims_supported": 10,\n'
            '    "reasoning": "Перечисление утверждений с поддержкой..."\n'
            '}},\n'
            '"answer_relevancy": {{\n'
            '    "score": 0.92,\n'
            '    "reasoning": "Краткое обоснование"\n'
            '}},\n'
            '"context_precision": {{\n'
            '    "score": 0.80,\n'
            '    "relevant_chunks": 4,\n'
            '    "total_chunks": 5,\n'
            '    "reasoning": "Анализ каждого чанка"\n'
            '}},\n'
            '"context_recall": {{\n'
            '    "score": 0.88,\n'
            '    "covered_claims": 7,\n'
            '    "total_gt_claims": 8,\n'
            '    "reasoning": "Анализ эталонного ответа"\n'
            '}}\n'
            '}}\n'
            "```"
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

        # 5. Инференс
        response = self.llm.invoke(formatted_prompt)
        print(f'RAG LLM RESPONSE {response}')
        # 6. Парсинг JSON
        # Ищем первый блок ```json ... ```
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
