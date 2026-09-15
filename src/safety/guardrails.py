"""Mesmo padrão de dosagem e frases do guardrail do notebook 10."""

import re

PADRAO_DOSE = r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|g|ml|mL|mg/m²|mg/m2)\b"
FRASES_PERIGOSAS = (
    "recomendo tomar", "deve tomar", "prescrevo", "a dose recomendada é",
)


def validar_seguranca(resposta: str) -> dict:
    possui_dose = bool(re.search(PADRAO_DOSE, resposta, flags=re.IGNORECASE))
    possui_prescricao = any(frase in resposta.lower() for frase in FRASES_PERIGOSAS)
    if possui_dose or possui_prescricao:
        return {
            "status": "bloqueado",
            "motivo_bloqueio": "Possível prescrição ou dosagem detectada.",
            "resposta_final": (
                "A resposta foi bloqueada pelo módulo de segurança. "
                "A definição de medicamentos ou doses deve ser realizada "
                "pelo profissional médico responsável."
            ),
        }
    return {"status": "aprovado", "motivo_bloqueio": "", "resposta_final": resposta}
