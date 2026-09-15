"""Carregamento e geração do nó LangGraph validado no notebook 10."""

from functools import lru_cache
from threading import Lock

from src import BASE_DIR

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = BASE_DIR / "models" / "qwen2.5-3b-medical-lora"
_GENERATION_LOCK = Lock()

SYSTEM_PROMPT = """
Você é um assistente clínico de apoio à decisão médica.

Utilize somente os dados do paciente e os protocolos institucionais fornecidos.

Regras obrigatórias:
- Não invente informações.
- Não emita diagnóstico definitivo.
- Não prescreva medicamentos.
- Não informe doses.
- Diferencie exames realizados de exames pendentes.
- Nunca atribua resultado a exame ainda pendente.
- A decisão final deve permanecer com o profissional médico responsável.
- Informe os protocolos utilizados como fonte.
- Não crie títulos ou seções sem conteúdo. Se não houver pontos adicionais de atenção, não escreva a seção 'Pontos de atenção'.
""".strip()


@lru_cache(maxsize=1)
def carregar_modelo():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not (ADAPTER_DIR / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Adapter LoRA não encontrado: {ADAPTER_DIR}")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()
    return tokenizer, model


def gerar_resposta(contexto_paciente: str, contexto_protocolos: str, pergunta: str) -> str:
    import torch

    # O recurso é compartilhado entre sessões; a geração é serializada.
    with _GENERATION_LOCK:
        tokenizer, model = carregar_modelo()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""
DADOS DO PACIENTE:
{contexto_paciente}

PROTOCOLOS:
{contexto_protocolos}

PERGUNTA:
{pergunta}
"""},
        ]
        texto = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(texto, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=400, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        novos_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(novos_tokens, skip_special_tokens=True)
