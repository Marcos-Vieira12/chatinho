#!/usr/bin/env python
# -- coding: utf-8 --

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

DEFAULT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
DEFAULT_VS_NAME = os.getenv("VECTOR_STORE_NAME", "rag_local")
DEFAULT_DOCS_DIR = os.getenv("DOCS_DIR", "./docs")
LOCAL_BASE = Path("rag_store")
CHAT_HISTORY = []

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
    """Traduz texto PT↔EN via Google Translate (deep-translator)."""
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

    pdfs = sorted(docs_dir.glob("*/.pdf"))
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

            ids = [f"{pdf.stem}p{page_num}{i}" for i in range(len(chunks))]
            metas = [
                {
                    "source": pdf.name,
                    "path": str(pdf),
                    "page": page_num,   
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

def extract_answer(resp):
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text.strip()

    try:
        for block in resp.output:
            if block.get("type") == "message":
                for item in block["content"]:
                    if item["type"] == "output_text":
                        return item["text"].strip()
    except Exception:
        pass

    return None

def ask(vs_name: str, question: str):
    
    global CHAT_HISTORY
    
    t_start = time.time()

    # --------------------------------------------------------------
    # Comando para limpar o histórico
    # --------------------------------------------------------------
    #isso pode desconsiderar, aqui era so pra debug, pra eu num ter que ficar so resetando
    #o server pra limpar o hisotico
    if question.strip().lower() in ['reset']:
        CHAT_HISTORY = []
        return "O histórico da conversa foi reiniciado"

    logger.info(f"→ Pergunta recebida: {question[:50]}...")

    t3 = time.time()
    local_path = ensure_local_dir(vs_name)
    collection = get_chroma_collection(vs_name)

    # Criar embedding
    emb_q = client.embeddings.create(
        model="text-embedding-3-large",
        input=question
    ).data[0].embedding
    time_spent = time.time() - t3
    logger.info(f"→ Geração de embedding: {time_spent:.2f}s")

    # Consultar ChromaDB
    t4 = time.time()
    results = collection.query(
        query_embeddings=[emb_q],
        n_results=4,
        include=["documents", "metadatas"]
    )
    time_spent = time.time() - t4
    logger.info(f"→ Consulta ao ChromaDB: {time_spent:.2f}s")

    # if not results["documents"][0]:
    #     return "Não encontrei informações relevantes sobre isso nos documentos de radiologia."

    # # Coletar fontes únicas
    # t_prompt = time.time() 
    # seen = set()
    # sources = []
    # for meta in results["metadatas"][0]: 
    #     src = f"{meta.get('source', 'desconhecido')} (p.{meta.get('page', '?')})"
    #     if src not in seen:
    #         seen.add(src)
    #         sources.append(src)

    # if results["documents"][0]:

    #     # Coletar fontes únicas
    #     t_prompt = time.time()
    #     seen = set()
    #     sources = []
    #     for meta in results["metadatas"][0]: 
    #         src = f"{meta.get('source', 'desconhecido')} (p.{meta.get('page', '?')})"
    #         if src not in seen:
    #             seen.add(src)
    #             sources.append(src)

    # context = "\n\n".join(results["documents"][0])

    if results["documents"][0]:
        seen = set()
        sources = []

        for meta in results["metadatas"][0]:
            src = f"{meta.get('source','desconhecido')} (p.{meta.get('page','?')})"
            if src not in seen:
                seen.add(src)
                sources.append(src)

        context = "\n\n".join(results["documents"][0])
        context += "\n\nFONTE:\n" + "\n".join(sources)
    else:
        context = ""

    # system_prompt = (
    #     "Você é um assistente altamente especializado e sempre responde em português.\n"
    #     "Use o histórico da conversa para manter coerência.\n"
    #     "se os trechos abaixo não forem relevantes para a resposta não use-os e não cite 'FONTE' e responda com base no seu conhecimento sobre o assunto.\n"
    #     "se os trechos forem relevantes siga a regra abaixo:\n"
    #     "1. Responda APENAS com base nos trechos abaixo e inclua:\n"
    #     "   FONTE:\n"
    #     "   → Apenas uma fonte por linha, sem duplicar.\n"
    #     "   Preserve medidas em milímetros exatamente como nos textos.\n\n"
    #     f"{context}\n\n"
    #     f"FONTE:\n{chr(10).join(sources)}\n"
    #     "obs: lembre-se, se vc não usou os trechos para gerar a resposta não os cite."
        
    # )

    # llm_input = [{"role": "system", "content": system_prompt}] + CHAT_HISTORY + [
    #     {"role": "user", "content": question}
    # ]
    

    system_prompt = (
        "Você é o DeuChat, o assistente virtual especializado em radiologia da DeuLaudo.\n" 
        "Você deve responder conforme (caso haja) exigências feitas nas perguntas acima ou (principalmente) na pergunta abaixo.\n"
        "se utilizar os trechos abaixo, é terminantemente proibido que você: especule, invente, preveja, ou afirme algo que não tenha certeza\n"
        "se os trechos abaixo não forem relevantes para a resposta não use-os e não cite 'FONTE' e responda com base no seu conhecimento sobre o assunto.\n"
        "obs: lembre-se, se vc não usou os trechos para gerar a resposta não os cite.\n"
        "REGRAS:\n"
        "1. Sempre deve responder em português.\n"
        "2. Responda com base nos trechos abaixo e inclua:\n"
        "   FONTE:\n"
        "   → Apenas uma fonte por linha, sem duplicar.\n"
        "   Preserve medidas em milímetros exatamente como nos textos.\n\n"
        "3. se você não utilizou as fotes para gerar a resposta não cite"
        #f"{context}\n\n"
        f"FONTE:\n{chr(10).join(sources)}\n"
        
    )

    # llm_input = CHAT_HISTORY + [{"role": "system", "content": system_prompt}] + [
    #     {"role": "user", "content": question}
    # ]
    # llm_input = [
    #     {"role": "system", "content": system_prompt},
    # ] + CHAT_HISTORY + [
    #     {"role": "user", "content": question}
    # ]

    llm_input = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": context},
    ] + CHAT_HISTORY + [
        {"role": "user", "content": question}
    ]

    # Chamada
    t5 = time.time()
    resp_final = client.responses.create(
        model=DEFAULT_MODEL, 
        temperature=0.0,    
        input=llm_input,
        max_output_tokens=4000
    )

    print("DEBUG:", resp_final.model_dump())

    answer = extract_answer(resp_final)
    if not answer:
        logger.error("→ Não consegui extrair texto do LLM.")
        return "Ocorreu um erro ao processar a resposta do modelo. Tente novamente."

    logger.info(f"→ Resposta GPT-4 obtida em {time.time() - t5:.2f}s")

    # ----------------------------------------------------------------------
    # Salva o histórico
    # ----------------------------------------------------------------------
    CHAT_HISTORY.append({"role": "user", "content": question})
    CHAT_HISTORY.append({"role": "assistant", "content": answer})

    if len(CHAT_HISTORY) > 10:
        CHAT_HISTORY[:] = CHAT_HISTORY[-10:]

    logger.info(f"→ Histórico salvo ({len(CHAT_HISTORY)} msgs)")

    return answer


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
