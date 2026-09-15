"""Fluxo do notebook 10, sem persistir estado entre pacientes."""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.chains.medical_rag import recuperar_protocolos
from src.database.database import buscar_paciente, formatar_paciente, validar_id
from src.llm.model import gerar_resposta
from src.safety.guardrails import validar_seguranca
from src.utils.logger import registrar_log


class EstadoAssistente(TypedDict, total=False):
    id_paciente: str
    pergunta: str
    paciente: dict
    contexto_paciente: str
    protocolos: list[dict]
    protocolos_recuperados: list[str]
    contexto_protocolos: str
    resposta: str
    resposta_final: str
    status: str
    motivo_bloqueio: str


def no_validar_entrada(state: EstadoAssistente):
    try:
        id_paciente = validar_id(state.get("id_paciente", ""))
    except ValueError as exc:
        return {"status": "erro", "resposta_final": str(exc)}
    pergunta = state.get("pergunta", "").strip()
    if not pergunta:
        return {"status": "erro", "resposta_final": "Pergunta não informada."}
    return {"id_paciente": id_paciente, "pergunta": pergunta, "status": "ok"}


def no_buscar_paciente(state: EstadoAssistente):
    paciente = buscar_paciente(state["id_paciente"])
    if paciente is None:
        return {"status": "erro", "resposta_final": "Paciente não encontrado."}
    return {"paciente": paciente, "contexto_paciente": formatar_paciente(paciente), "status": "ok"}


def no_buscar_protocolos(state: EstadoAssistente):
    docs = recuperar_protocolos(state["pergunta"], state["contexto_paciente"])
    return {
        "protocolos_recuperados": [doc.metadata["id"] for doc in docs],
        "contexto_protocolos": "\n\n".join(doc.page_content for doc in docs),
        "protocolos": [{**doc.metadata, "conteudo": doc.page_content} for doc in docs],
    }


def no_gerar_resposta(state: EstadoAssistente):
    return {"resposta": gerar_resposta(
        state["contexto_paciente"], state["contexto_protocolos"], state["pergunta"],
    )}


def no_validar_seguranca(state: EstadoAssistente):
    resultado = validar_seguranca(state["resposta"])
    # A resposta bloqueada não segue no estado retornado para a interface.
    return {**resultado, "resposta": "" if resultado["status"] == "bloqueado" else state["resposta"]}


def no_registrar_log(state: EstadoAssistente):
    registrar_log(state)
    return {}


def criar_grafo():
    builder = StateGraph(EstadoAssistente)
    for nome, funcao in (
        ("validar_entrada", no_validar_entrada),
        ("buscar_paciente", no_buscar_paciente),
        ("buscar_protocolos", no_buscar_protocolos),
        ("gerar_resposta", no_gerar_resposta),
        ("validar_seguranca", no_validar_seguranca),
        ("registrar_log", no_registrar_log),
    ):
        builder.add_node(nome, funcao)
    builder.add_edge(START, "validar_entrada")
    builder.add_conditional_edges(
        "validar_entrada",
        lambda state: "log" if state.get("status") == "erro" else "paciente",
        {"paciente": "buscar_paciente", "log": "registrar_log"},
    )
    builder.add_conditional_edges(
        "buscar_paciente",
        lambda state: "log" if state.get("status") == "erro" else "protocolos",
        {"protocolos": "buscar_protocolos", "log": "registrar_log"},
    )
    builder.add_edge("buscar_protocolos", "gerar_resposta")
    builder.add_edge("gerar_resposta", "validar_seguranca")
    builder.add_edge("validar_seguranca", "registrar_log")
    builder.add_edge("registrar_log", END)
    return builder.compile()
