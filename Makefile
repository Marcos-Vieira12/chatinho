
# --- Configuração ---
SHELL := /bin/bash

# Ambiente virtual
VENV_NAME = .venv
PYTHON = $(VENV_NAME)/bin/python
PIP = $(VENV_NAME)/bin/pip

# Configs do Servidor
BACKEND_APP = backend.cloud_rag_server:app
HOST = 127.0.0.1
PORT = 8000

include backend/.env
export

.PHONY: help install run index clean drop

help:
	@echo "Comandos disponíveis:"
	@echo "  make install  - Cria o ambiente virtual e instala dependências"
	@echo "  make index    - (Re)Indexa os PDFs localmente via ChromaDB"
	@echo "  make run      - Inicia o servidor (API + Frontend) em http://$(HOST):$(PORT)"
	@echo "  make drop     - Remove a base vetorial local"
	@echo "  make clean    - Remove o ambiente virtual e caches"

install: $(VENV_NAME)/bin/activate

$(VENV_NAME)/bin/activate: backend/requirements.txt
	@echo "--- Criando ambiente virtual em '$(VENV_NAME)'..."
	python3 -m venv $(VENV_NAME)
	@echo "--- Instalando dependências de 'backend/requirements.txt'..."
	$(PIP) install --upgrade pip
	$(PIP) install -r backend/requirements.txt
	@echo "--- Instalação concluída! ---"
	@echo "Execute 'source $(VENV_NAME)/bin/activate' para ativar manualmente, se desejar."

index: $(VENV_NAME)/bin/activate
	@echo "--- Iniciando indexação LOCAL dos PDFs em '$(DOCS_DIR)' ---"
	@echo "Base local: rag_store/$(VECTOR_STORE_NAME)"
	# Executa o novo RAG local com ChromaDB
	$(PYTHON) backend/cloud_rag_cli.py index --docs $(DOCS_DIR) --vs-name $(VECTOR_STORE_NAME)
	@echo "--- Indexação LOCAL concluída ---"

run: $(VENV_NAME)/bin/activate
	@echo "--- Iniciando o servidor UNIFICADO (API + Frontend) ---"
	@echo "Acesse: http://$(HOST):$(PORT)"
	@echo "API docs: http://$(HOST):$(PORT)/docs"
	# Roda o uvicorn A PARTIR DA RAIZ
	$(PYTHON) -m uvicorn $(BACKEND_APP) --host $(HOST) --port $(PORT) --reload --reload-dir backend

drop: $(VENV_NAME)/bin/activate
	@echo "--- Removendo a base vetorial local '$(VECTOR_STORE_NAME)' ---"
	$(PYTHON) backend/cloud_rag_cli.py drop --vs-name $(VECTOR_STORE_NAME)
	@echo "--- Base local apagada ---"

clean:
	@echo "--- Limpando ambiente virtual e caches ---"
	rm -rf $(VENV_NAME)
	rm -rf backend/__pycache__
	rm -rf rag_store
	find . -name "*.pyc" -exec rm -f {} \;
	@echo "--- Limpeza concluída ---"
