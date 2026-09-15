"""Registro UTF-8 sem reconfigurar os loggers do Streamlit."""

import json
import logging
from functools import lru_cache

from src import BASE_DIR

LOG_PATH = BASE_DIR / "logs" / "assistant.log"


@lru_cache(maxsize=1)
def configurar_logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("assistente_clinico")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    return logger


def registrar_log(state: dict) -> None:
    registro = {
        "paciente": state.get("id_paciente"),
        "pergunta": state.get("pergunta"),
        "protocolos": state.get("protocolos_recuperados", []),
        "status": state.get("status"),
        "bloqueio": state.get("motivo_bloqueio") or "nenhum",
    }
    configurar_logger().info(json.dumps(registro, ensure_ascii=False))
