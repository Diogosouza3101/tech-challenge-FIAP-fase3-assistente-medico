"""Busca semântica e inclusão obrigatória de protocolos do notebook 10."""

import json
from functools import lru_cache

from langchain_core.documents import Document

from src import BASE_DIR

PROTOCOLOS_PATH = BASE_DIR / "data" / "raw" / "protocolos_medicos.json"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def carregar_documentos() -> list[Document]:
    protocolos = json.loads(PROTOCOLOS_PATH.read_text(encoding="utf-8"))
    return [
        Document(
            page_content=(
                f"Protocolo: {p['id']}\n"
                f"Título: {p['titulo']}\n"
                f"Conteúdo: {p['conteudo']}"
            ),
            metadata={
                "id": p["id"], "titulo": p["titulo"],
                "fonte": PROTOCOLOS_PATH.relative_to(BASE_DIR).as_posix(),
            },
        )
        for p in protocolos
    ]


@lru_cache(maxsize=1)
def carregar_rag():
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    documentos = carregar_documentos()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(documentos, embeddings)
    return documentos, vectorstore.as_retriever(search_kwargs={"k": 3})


def recuperar_protocolos(pergunta: str, contexto_paciente: str) -> list[Document]:
    documentos, retriever = carregar_rag()
    docs_rag = retriever.invoke(pergunta + "\n" + contexto_paciente)
    ids_selecionados = {doc.metadata["id"] for doc in docs_rag}
    if (
        "EXAME PENDENTE:" in contexto_paciente
        and "EXAME PENDENTE: Nenhum" not in contexto_paciente
    ):
        ids_selecionados.add("PROTO-003")
    ids_selecionados.update({"PROTO-004", "PROTO-005"})
    return [doc for doc in documentos if doc.metadata["id"] in ids_selecionados]
