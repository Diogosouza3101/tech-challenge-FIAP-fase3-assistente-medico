"""Execute na raiz: streamlit run app/app.py."""

import re
import sys
from pathlib import Path

import streamlit as st

# O Streamlit inclui app/ no sys.path; os módulos src ficam na raiz.
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.graph.medical_graph import criar_grafo

st.set_page_config(page_title="Assistente Clínico com IA")
st.title("Assistente Clínico com IA")
st.warning("Esta ferramenta não substitui a avaliação médica. A decisão final cabe ao profissional médico responsável.")


@st.cache_resource
def carregar_grafo():
    return criar_grafo()


def limpar_resposta(texto: str) -> str:
    # Remove somente o título final vazio, mesmo com Markdown incompleto.
    texto = re.sub(
        r"(?:^|\n)\s*(?:#{1,6}\s*)?\*{0,2}\s*"
        r"Pontos\s+de\s+aten[cç][aã]o\s*\*{0,2}\s*:?\s*$",
        "",
        texto.rstrip(),
        flags=re.IGNORECASE,
    )
    return texto.rstrip()


id_paciente = st.text_input("ID do paciente", placeholder="PAC001")
pergunta = st.text_area("Pergunta")

if st.button("Analisar", type="primary"):
    try:
        with st.spinner("Analisando os dados e protocolos..."):
            resultado = carregar_grafo().invoke({"id_paciente": id_paciente, "pergunta": pergunta})
    except Exception:
        st.error("Não foi possível concluir a análise. Verifique as dependências, o banco, os protocolos e a disponibilidade do modelo no ambiente.")
    else:
        if resultado.get("contexto_paciente"):
            st.subheader("Dados utilizados")
            st.text(resultado["contexto_paciente"])
        if resultado.get("protocolos"):
            st.subheader("Protocolos e fontes recuperados")
            for protocolo in resultado["protocolos"]:
                with st.expander(f"{protocolo['id']} — {protocolo['titulo']}"):
                    st.text(protocolo["conteudo"])
                    st.caption(f"Fonte: {protocolo['fonte']}")
        st.subheader("Resposta final")
        resposta_limpa = limpar_resposta(resultado["resposta_final"])
        st.markdown(resposta_limpa)
        st.subheader("Status de segurança")
        status = resultado["status"]
        if status == "aprovado":
            st.success("Aprovado")
        elif status == "bloqueado":
            st.error(f"Bloqueado: {resultado['motivo_bloqueio']}")
        else:
            st.info("Não avaliado: entrada inválida ou paciente não encontrado.")
