import traceback
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import subprocess, shlex
from pathlib import Path
from backend.cloud_rag_cli import ask as rag_ask

# --------------------------------------------------------------------
app = FastAPI(title="DeuChat - RAG Local + Frontend")

origins = [
    "https://chatinho-6c7o.onrender.com",  # seu frontend hospedado
    "http://localhost:5173",                # opcional, para testes locais
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# --------------------------------------------------------------------
@app.post("/ask")
async def ask_endpoint(request: Request):
    try:
        data = await request.json()
        q = data.get("q")
        vs_name = data.get("vs_name", "rag_local")

        if not q:
            return JSONResponse({"error": "Campo 'q' é obrigatório."}, status_code=400)

        # --- chama diretamente a função Python ---
        resposta = rag_ask(vs_name, q)

        return JSONResponse({"response": resposta})

    except Exception as e:
        print("[ERRO no /ask]", e)
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
    
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
