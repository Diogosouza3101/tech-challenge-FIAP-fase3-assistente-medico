"""Consulta fixa, restrita ao ID solicitado, conforme notebooks 08 e 10."""

import re
import sqlite3
from contextlib import closing
from pathlib import Path

from src import BASE_DIR

DB_PATH = BASE_DIR / "data" / "database" / "hospital.db"


def validar_id(id_paciente: str) -> str:
    id_paciente = id_paciente.upper().strip()
    if not re.fullmatch(r"PAC\d{3}", id_paciente):
        raise ValueError("ID de paciente inválido.")
    return id_paciente


def buscar_paciente(id_paciente: str, db_path: Path = DB_PATH) -> dict | None:
    id_paciente = validar_id(id_paciente)
    # mode=ro evita criar ou modificar o banco validado nos notebooks.
    with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        registro = conn.execute(
            """SELECT id_paciente, idade, sexo, historico_familiar,
                      resultado_exame, exames_pendentes, status
               FROM pacientes WHERE id_paciente = ?""",
            (id_paciente,),
        ).fetchone()
    return dict(registro) if registro is not None else None


def formatar_paciente(paciente: dict) -> str:
    return f"""ID DO PACIENTE: {paciente['id_paciente']}
IDADE: {paciente['idade']}
SEXO: {paciente['sexo']}
HISTÓRICO FAMILIAR: {paciente['historico_familiar']}
RESULTADO DO EXAME JÁ REALIZADO: {paciente['resultado_exame']}
EXAME PENDENTE: {paciente['exames_pendentes']}
STATUS ATUAL: {paciente['status']}"""
