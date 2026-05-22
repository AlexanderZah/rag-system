from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Ты — помощник для Финансовой системы РЦК.
Отвечай **только** на русском языке, кратко, по делу и структурировано.
Используй **только** информацию из предоставленного контекста.
Для каждого факта/шага указывай источник в формате [1], [2] и т.д.
Если информации недостаточно — честно скажи, чего именно не хватает.
Не придумывай факты."""


class EnhancedGenerator:
    def __init__(self, model_name="Qwen/Qwen2.5-14B-Instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        logger.info(f"Loading model {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Настройка 4-битной квантизации через BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("Model loaded successfully.")

    def generate(
        self,
        query: str,
        context_str: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        show_prompt: bool = False
    ) -> str:

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""Контекст:
{context_str}

Вопрос: {query}"""}
        ]

        # 3. Применяем chat template
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        if show_prompt:
            logger.info("=== PROMPT ===\n%s\n==============", prompt_text)

        # 4. Токенизируем и генерируем
        model_inputs = self.tokenizer(
            [prompt_text],
            return_tensors="pt",
            truncation=True,
            max_length=32000
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][model_inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True).strip()

        return response
