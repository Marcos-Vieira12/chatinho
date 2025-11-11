#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
cloud_rag_cli_local.py — RAG com ChromaDB
Uso:
  python cloud_rag_cli.py index --docs ./docs --vs-name rag_local
  python cloud_rag_cli.py list  --vs-name rag_local
  python cloud_rag_cli.py ask   --vs-name rag_local "Pergunta em PT ou EN"
  python cloud_rag_cli.py drop  --vs-name rag_local
"""

import argparse
import os
import sys
import json
import shutil
import time
import logging
import chromadb
import traceback
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
from langdetect import detect
from openai import OpenAI
from chromadb.config import Settings
from pypdf import PdfReader
from chromadb import PersistentClient
from deep_translator import GoogleTranslator

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
dotenv_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=dotenv_path, override=True)

DEFAULT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
DEFAULT_VS_NAME = os.getenv("VECTOR_STORE_NAME", "rag_local")
DEFAULT_DOCS_DIR = os.getenv("DOCS_DIR", "./docs")
LOCAL_BASE = Path("rag_store")

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("[err] OPENAI_API_KEY não encontrado no .env")
    sys.exit(1)

logging.basicConfig(
    filename='log.txt',
    filemode='a',
    format='%(asctime)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger()
client = OpenAI(api_key=API_KEY)

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def ensure_local_dir(vs_name: str) -> Path:
    path = LOCAL_BASE / vs_name
    path.mkdir(parents=True, exist_ok=True)
    return path

def log(local_path: Path, msg: str):
    with open(local_path / "logs.txt", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

def extract_text_from_pdf(pdf_path: Path) -> list[tuple[str, int]]:
    """Extrai texto de cada página e retorna uma lista de (texto, número_da_página)."""
    pages = []
    try:
        reader = PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append((text, i + 1))  # página começa em 1
    except Exception as e:
        print(f"[warn] Falha ao ler {pdf_path.name}: {e}")
    return pages

def chunk_text(text: str, max_chars: int = 1000):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def auto_translate(text: str, target_lang: str = "en"):
    """Traduz texto PT↔EN via Google Translate (via deep-translator)."""
    try:
        if target_lang == "en":
            return GoogleTranslator(source='pt', target='en').translate(text)
        else:
            return GoogleTranslator(source='en', target='pt').translate(text)
    except Exception as e:
        print(f"[warn] Falha na tradução: {e}")
        return text

# -------------------------------------------------------------------
# ChromaDB
# -------------------------------------------------------------------
def get_chroma_collection(vs_name: str):
    persist_dir = str(LOCAL_BASE / vs_name)
    client_chroma = PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    return client_chroma.get_or_create_collection(name=vs_name)

# -------------------------------------------------------------------
# Indexação
# -------------------------------------------------------------------
def index_pdfs(docs_dir: Path, vs_name: str):
    local_path = ensure_local_dir(vs_name)
    collection = get_chroma_collection(vs_name)

    pdfs = sorted(docs_dir.glob("**/*.pdf"))
    if not pdfs:
        print(f"[warn] Nenhum PDF encontrado em {docs_dir}")
        return

    for pdf in tqdm(pdfs, desc="Indexando PDFs"):
        pages = extract_text_from_pdf(pdf)
        if not pages:
            continue

        for page_text, page_num in pages:
            chunks = chunk_text(page_text)
            if not chunks:
                continue

            # Cria embeddings por página
            resp = client.embeddings.create(
                model="text-embedding-3-large",
                input=chunks
            )
            embeds = [d.embedding for d in resp.data]

            ids = [f"{pdf.stem}_p{page_num}_{i}" for i in range(len(chunks))]
            metas = [
                {
                    "source": pdf.name,
                    "path": str(pdf),
                    "page": page_num,   # 🔹 nova informação!
                    "chunk": i
                }
                for i in range(len(chunks))
            ]

            collection.add(
                documents=chunks,
                embeddings=embeds,
                ids=ids,
                metadatas=metas
            )

        print(f"[ok] {pdf.name}: {len(pages)} páginas indexadas.")

    log(local_path, f"Indexados {len(pdfs)} PDFs.")
    print(f"[ok] Indexação concluída: {len(pdfs)} PDFs.")

# -------------------------------------------------------------------
# Listagem
# -------------------------------------------------------------------
def list_docs(vs_name: str):
    collection = get_chroma_collection(vs_name)
    count = collection.count()
    print(f"[info] {count} chunks armazenados em {vs_name}")

# -------------------------------------------------------------------
# Pergunta
# -------------------------------------------------------------------
def ask(vs_name: str, question: str):
    """
    Consulta RAG local se a pergunta for sobre radiologia clínica, 
    caso contrário, responde normalmente.
    """
    
    t_start = time.time()

    # ----------------------------------------------------------------------
    #Detecção de idioma
    # ----------------------------------------------------------------------
    t0 = time.time()
    try:
        lang = detect(question)
    except Exception:
        lang = "en"
    time_spent = time.time() - t0
    logger.info(f"→ Detecção de idioma: {lang} ({time_spent:.2f}s)")

    # ----------------------------------------------------------------------
    #Tradução PT→EN (se necessário)
    # ----------------------------------------------------------------------
    t1 = time.time()
    q_en = auto_translate(question, "en") if lang != "en" else question
    time_spent = time.time() - t1
    logger.info(f"→ Tradução PT→EN: ({time_spent:.2f}s)")

    t2 = time.time()
    
    # [NOVO PROMPT v3 - Mais inteligente]
    classification_prompt = (
        "Sua tarefa é classificar o texto do usuário em uma de duas categorias: 'RADIOLOGIA' ou 'GERAL'.\n"
        "Responda APENAS com a palavra da categoria, e nada mais.\n\n"
        "CATEGORIAS:\n"
        "1. RADIOLOGIA: APENAS perguntas que solicitam interpretação, definição de achados de exames (TC, RM, Raio-X), ou que contenham termos médicos e detalhes clínicos de um laudo.\n"
        "   Exemplos de RADIOLOGIA: 'O que é um nódulo pulmonar de 5mm?', 'Biótipo brevilíneo afeta o exame?', 'interprete esta tomografia', 'o que significa hipoatenuante no fígado?'\n"
        "\n"
        "2. GERAL: Qualquer outra coisa. Isso inclui saudações, perguntas genéricas, E TAMBÉM conversas 'sobre' o tópico de radiologia que NÃO contenham detalhes clínicos (meta-conversa).\n"
        "   Exemplos GERAIS (Triviais): 'oi', 'tudo bem?', 'qual a capital da França?', 'explique este código python'\n"
        "   Exemplos GERAIS (Meta-conversa): 'gostaria de fazer uma pergunta sobre radiologia', 'você entende de laudos?', 'o que você sabe sobre radiologia?', 'posso te enviar um exame?'\n"
        "\n"
        "--- INÍCIO DO TEXTO DO USUÁRIO ---\n"
        f"{q_en}\n"
        "--- FIM DO TEXTO DO USUÁRIO ---\n"
        "\n"
        "CATEGORIA (apenas uma palavra):"
    )
    
    resp_classification = client.responses.create(
        model=DEFAULT_MODEL,
        input=[{"role": "user", "content": classification_prompt}],
        temperature=0.0,
        max_output_tokens=16
    )
    
    classification = resp_classification.output_text.strip().upper()
    time_spent = time.time() - t2
    logger.info(f"→ Classificação: ({time_spent:.2f}s)")

    # ----------------------------------------------------------------------
    # Roteamento: RAG (Radiologia)
    # ----------------------------------------------------------------------
    
    if "RADIOLOGIA" in classification:
        t3 = time.time()
        local_path = ensure_local_dir(vs_name)
        collection = get_chroma_collection(vs_name)

        # Criar embedding
        emb_q = client.embeddings.create(
            model="text-embedding-3-large",
            input=q_en
        ).data[0].embedding
        time_spent = time.time() - t3 # O tempo de embedding começa aqui (t3)
        logger.info(f"→ Geração de embedding: {time_spent:.2f}s")

        # Consultar ChromaDB
        t4 = time.time()
        results = collection.query(
            query_embeddings=[emb_q],
            n_results=7,
            include=["documents", "metadatas"]
        )
        time_spent = time.time() - t4
        logger.info(f"→ Consulta ao ChromaDB: {time_spent:.2f}s")

        if not results["documents"][0]:
            return "Não encontrei informações relevantes sobre isso nos documentos de radiologia."

        # Coletar fontes únicas
        t_prompt = time.time() # Novo timer para montagem do prompt
        seen = set()
        sources = []
        for meta in results["metadatas"][0]: 
            src = f"{meta.get('source', 'desconhecido')} (p.{meta.get('page', '?')})"
            if src not in seen:
                seen.add(src)
                sources.append(src)

        context = "\n\n".join(results["documents"][0])
        prompt = (
            "Você é um assistente médico especializado em radiologia e sempre deve responder em português.\n"
            "REGRAS:\n"
            "1. Responda APENAS com base nos trechos abaixo e inclua:\n"
            "   FONTE:\n"
            "   → Apenas uma fonte por linha, sem duplicar.\n"
            "   Preserve medidas em milímetros exatamente como nos textos.\n\n"
            f"{context}\n\n"
            f"Pergunta: {q_en}\n\n"
            f"FONTE:\n{chr(10).join(sources)}"
        )
        time_spent = time.time() - t_prompt # Usando o novo timer
        logger.info(f"→ montagem do prompt: {time_spent:.2f}s")

        t5 = time.time()
        resp_final = client.responses.create(
            model=DEFAULT_MODEL,
            input=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_output_tokens=800
        )
        answer_en = resp_final.output_text.strip()
        time_spent = time.time() - t5
        logger.info(f"→ Resposta GPT (Radiologia): {time_spent:.2f}s")

    else:
        # Não é radiologia
        t_geral = time.time()
        
        general_prompt = (
            "Você é um assistente prestativo. Responda a pergunta abaixo em português de forma completa e útil.\n\n"
            f"Pergunta: {q_en}"
        )
        resp_general = client.responses.create(
            model=DEFAULT_MODEL,
            input=[{"role": "user", "content": general_prompt}],
            temperature=0.7, 
            max_output_tokens=800
        )
        answer_en = resp_general.output_text.strip()
        time_spent = time.time() - t_geral
        logger.info(f"→ Resposta GPT (Geral): {time_spent:.2f}s")

    total = time.time() - t_start
    logger.info(f"# Tempo total: {total:.2f}s\n") # Adiciona uma linha em branco no log
    
    return answer_en

# -------------------------------------------------------------------
# Drop
# -------------------------------------------------------------------
def drop_store(vs_name: str):
    local_path = LOCAL_BASE / vs_name
    if local_path.exists():
        shutil.rmtree(local_path,ignore_errors=True)
        print(f"[ok] Base local removida: {local_path}")
    else:
        print("[info] Nenhuma base local encontrada.")

# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="RAG local (ChromaDB + GPT)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_idx = sub.add_parser("index", help="Indexar PDFs localmente")
    p_idx.add_argument("--docs", default=DEFAULT_DOCS_DIR)
    p_idx.add_argument("--vs-name", default=DEFAULT_VS_NAME)
    p_idx.set_defaults(func="index")

    p_list = sub.add_parser("list", help="Listar documentos")
    p_list.add_argument("--vs-name", default=DEFAULT_VS_NAME)
    p_list.set_defaults(func="list")

    p_ask = sub.add_parser("ask", help="Perguntar ao RAG local")
    p_ask.add_argument("--vs-name", default=DEFAULT_VS_NAME)
    p_ask.add_argument("question", nargs="+")
    p_ask.set_defaults(func="ask")

    p_drop = sub.add_parser("drop", help="Apagar base local")
    p_drop.add_argument("--vs-name", default=DEFAULT_VS_NAME)
    p_drop.set_defaults(func="drop")

    args = ap.parse_args()

    if args.func == "index":
        index_pdfs(Path(args.docs).resolve(), args.vs_name)
    elif args.func == "list":
        list_docs(args.vs_name)
    elif args.func == "ask":
        q = " ".join(args.question)
        ask(args.vs_name, q)
    elif args.func == "drop":
        drop_store(args.vs_name)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
