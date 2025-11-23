
"""
cloud_rag_cli_local.py — RAG com ChromaDB + BGE-M3 + PubMedBERT reranker
"""

import argparse
import os
import sys
import json
import shutil
import time
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
from langdetect import detect
from openai import OpenAI
from chromadb.config import Settings
from chromadb import PersistentClient
from pypdf import PdfReader
from deep_translator import GoogleTranslator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import SentenceTransformer
import torch



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[info] Device for models: {DEVICE}")

bge_model = BGEM3FlagModel(model_name_or_path="BAAI/bge-m3", device=DEVICE)
#bge = SentenceTransformer("BAAI/bge-m3")


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

vs_name = "rag100"
persist_dir = str(LOCAL_BASE / vs_name)
client_chroma = PersistentClient(
    path=persist_dir,
    settings=Settings(anonymized_telemetry=False)
)
collection = client_chroma.get_or_create_collection(name=vs_name)

logger = logging.getLogger()
client = OpenAI(api_key=API_KEY)

# -------------------------------------------------------------------
# PubMedBERT Reranker
# -------------------------------------------------------------------
# TOKENIZER = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
# MODEL = AutoModelForSequenceClassification.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
TOKENIZER = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
MODEL = AutoModelForSequenceClassification.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
MODEL.to(DEVICE)
MODEL.eval()

# def rerank(question, docs, top_k=4):

#     scores = []
#     inputs = TOKENIZER(
#         [question]*len(docs),
#         docs,
#         padding=True,
#         truncation=True,
#         return_tensors="pt"
#     )
#     with torch.no_grad():
#         logits = MODEL(**inputs).logits

#         rel_scores = F.softmax(logits, dim=1)[:, 1] if logits.shape[1] > 1 else logits[:,0]
#         scores = rel_scores.tolist()
#     doc_scores = list(zip(docs, scores))
#     doc_scores.sort(key=lambda x: x[1], reverse=True)
#     return [d for d, s in doc_scores[:top_k]]
def rerank(question, docs, top_k=4, batch_size=16, max_length=256):

    if not docs:
        return []

    pairs = [(question, d) for d in docs]
    scores = []
    MODEL.eval()
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            queries = [q for q, _ in batch]
            passages = [p for _, p in batch]
            inputs = TOKENIZER(
                queries,
                passages,
                padding="longest",         
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            if DEVICE.startswith("cuda"):
                with torch.cuda.amp.autocast():
                    logits = MODEL(**inputs).logits
            else:
                logits = MODEL(**inputs).logits
            if logits.shape[1] > 1:
                rel_scores = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
            else:
                rel_scores = logits[:, 0].detach().cpu().tolist()
            scores.extend(rel_scores)

    doc_scores = list(zip(docs, scores, range(len(docs))))
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    top = doc_scores[:top_k]
    return top

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
    pages = []
    try:
        reader = PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append((text, i + 1))
    except Exception as e:
        print(f"[warn] Falha ao ler {pdf_path.name}: {e}")
    return pages

def chunk_text(text: str, max_chars: int = 1000):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def auto_translate(text: str, target_lang: str = "en"):
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

            # Cria embeddings BGE-M3
            resp = bge_model.encode(chunks)
            embeds = resp['dense_vecs']
            embeds_list = embeds.tolist()

            ids = [f"{pdf.stem}_p{page_num}_{i}" for i in range(len(chunks))]
            metas = [
                {"source": pdf.name, "path": str(pdf), "page": page_num, "chunk": i}
                for i in range(len(chunks))
            ]

            collection.add(
                documents=chunks,
                embeddings=embeds_list,
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
# Perguntas / RAG
# -------------------------------------------------------------------
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

# def ask(vs_name: str, question: str):
#     global CHAT_HISTORY
#     t_start = time.time()

#     if question.strip().lower() in ['reset']:
#         CHAT_HISTORY = []
#         return "O histórico da conversa foi reiniciado"

#     logger.info(f"→ Pergunta recebida: {question[:50]}...")

#     local_path = ensure_local_dir(vs_name)
#     t1 = time.time()

#     t1 = time.time()
#     resp = bge_model.encode([question])
#     emb_q = resp["dense_vecs"][0].tolist()
#     print(f"→ embedding da pergunta: {time.time()-t1:.2f}s")

#     t1 = time.time()
#     results = collection.query(
#         query_embeddings=[emb_q],
#         n_results=20,
#         include=["documents", "metadatas"]
#     )
#     print(f"→ get chromaDB: {time.time()-t1:.2f}s")

#     t1 = time.time()
#     retrieved_docs = results["documents"][0] if results["documents"][0] else []
#     top_docs = rerank(question, retrieved_docs, top_k=4)
#     print(f"→ Rerank dos k_top resultados mais importantes : {time.time()-t1:.2f}s")

#     t1 = time.time()
#     if top_docs:
#         seen = set()
#         sources = []
#         for i, doc_text in enumerate(top_docs):
#             meta = results["metadatas"][0][i]
#             src = f"{meta.get('source','desconhecido')} (p.{meta.get('page','?')})"
#             if src not in seen:
#                 seen.add(src)
#                 sources.append(src)
#         context = "\n\n".join(top_docs)
#         context += "\n\nFONTE:\n" + "\n".join(sources)
#     else:
#         context = ""

#     system_prompt = (
#         "Você é o DeuChat, o assistente virtual especializado em radiologia da DeuLaudo.\n" 
#         "Você deve responder conforme (caso haja) exigências feitas nas perguntas acima ou (principalmente) na pergunta abaixo.\n"
#         "se utilizar os trechos abaixo, é terminantemente proibido que você: especule, invente, preveja, ou afirme algo que não tenha certeza\n"
#         "se os trechos abaixo não forem relevantes para a resposta não use-os e não cite 'FONTE' e responda com base no seu conhecimento sobre o assunto.\n"
#         "obs: lembre-se, se vc não usou os trechos para gerar a resposta não os cite.\n"
#         "REGRAS:\n"
#         "1. Sempre deve responder em português.\n"
#         "2. Responda com base nos trechos abaixo e inclua:\n"
#         "   FONTE:\n"
#         "   → Apenas uma fonte por linha, sem duplicar.\n"
#         "   Preserve medidas em milímetros exatamente como nos textos.\n\n"
#         "3. se você não utilizou as fotes para gerar a resposta não cite\n"
#         f"FONTE:\n{chr(10).join(sources)}\n"
#     )

#     llm_input = [
#         {"role": "system", "content": system_prompt},
#         {"role": "assistant", "content": context},
#     ] + CHAT_HISTORY + [
#         {"role": "user", "content": question}
#     ]



#     resp_final = client.responses.create(
#         model=DEFAULT_MODEL,
#         temperature=0.0,
#         input=llm_input,
#         max_output_tokens=4000
#     )

#     answer = extract_answer(resp_final)
#     print(f"→ Resposta da OpenAI : {time.time()-t1:.2f}s")
#     if not answer:
#         logger.error("→ Não consegui extrair texto do LLM.")
#         return "Ocorreu um erro ao processar a resposta do modelo. Tente novamente."

#     CHAT_HISTORY.append({"role": "user", "content": question})
#     CHAT_HISTORY.append({"role": "assistant", "content": answer})
#     if len(CHAT_HISTORY) > 10:
#         CHAT_HISTORY[:] = CHAT_HISTORY[-10:]

#     logger.info(f"→ Histórico salvo ({len(CHAT_HISTORY)} msgs)")

#     return answer

def ask(vs_name: str, question: str):
    global CHAT_HISTORY
    t_start = time.time()

    if question.strip().lower() in ['reset']:
        CHAT_HISTORY = []
        return "O histórico da conversa foi reiniciado"

    logger.info(f"→ Pergunta recebida: {question[:50]}...")
    local_path = ensure_local_dir(vs_name)

    t1 = time.time()
    resp = bge_model.encode([question])       
    emb_q = resp["dense_vecs"][0].tolist()
    print(f"→ embedding da pergunta: {time.time()-t1:.2f}s")

    # 2) Query no Chroma
    t1 = time.time()
    results = collection.query(
        query_embeddings=[emb_q],
        n_results=20,
        include=["documents", "metadatas", "distances"]
    )
    print(f"→ get chromaDB: {time.time()-t1:.2f}s")


    retrieved_docs = results.get("documents", [[]])[0]
    retrieved_meta = results.get("metadatas", [[]])[0]

    t1 = time.time()
    reranked = rerank(question, retrieved_docs, top_k=4)
    print(f"→ Rerank dos k_top resultados mais importantes : {time.time()-t1:.2f}s")

    if reranked:
        top_docs = []
        sources = []
        for doc_text, score, orig_idx in reranked:
            meta = retrieved_meta[orig_idx]
            src = f"{meta.get('source','desconhecido')} (p.{meta.get('page','?')})"
            if src not in sources:
                sources.append(src)
            top_docs.append(doc_text)
        context = "\n\n".join(top_docs)
        context += "\n\nFONTE:\n" + "\n".join(sources)
    else:
        context = ""

    print(context)

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
        "3. se você não utilizou as fontes para gerar a resposta não cite\n"
        f"FONTE:\n{chr(10).join(sources) if sources else ''}\n"
    )

    llm_input = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": context},
    ] + CHAT_HISTORY + [
        {"role": "user", "content": question}
    ]

    print(llm_input)

    t1 = time.time()
    resp_final = client.responses.create(
        model=DEFAULT_MODEL,
        temperature=0.0,
        input=llm_input,
        max_output_tokens=4000
    )
    print(f"→ Resposta da OpenAI : {time.time()-t1:.2f}s")

    answer = extract_answer(resp_final)
    if not answer:
        logger.error("→ Não consegui extrair texto do LLM.")
        return "Ocorreu um erro ao processar a resposta do modelo. Tente novamente."

    # 8) histórico
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
        shutil.rmtree(local_path, ignore_errors=True)
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
        print(ask(args.vs_name, q))
    elif args.func == "drop":
        drop_store(args.vs_name)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
