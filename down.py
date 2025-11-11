# import os
# import time
# import requests
# from requests.exceptions import RequestException
# from Bio import Entrez
# from bs4 import BeautifulSoup 
# from urllib.parse import urljoin 
# import glob 

# # --- Bibliotecas do Selenium ---
# from selenium import webdriver
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from webdriver_manager.chrome import ChromeDriverManager
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC

# # ---------------------------------------------------------------
# # CONFIGURAÇÕES
# # ---------------------------------------------------------------
# QUERY = '((("Radiology"[MeSH Major Topic]) OR (jsubsetr[text]))) AND ("2023/01/01"[Date - Publication] : "2023/12/31"[Date - Publication])'
# MAX_ARTICLES = 400 
# DOWNLOAD_FOLDER = "Radiology_PDFs"
# YOUR_EMAIL = "jadriannassilva@gmail.com" # SEU EMAIL

# # ---------------------------------------------------------------
# # FUNÇÃO setup_environment (Idêntica)
# # ---------------------------------------------------------------
# def setup_environment():
#     if not os.path.exists(DOWNLOAD_FOLDER):
#         os.makedirs(DOWNLOAD_FOLDER)
#         print(f"Pasta '{DOWNLOAD_FOLDER}' criada.")

# # ---------------------------------------------------------------
# # FUNÇÃO download_pdf (Híbrido v6.3 - Esperando o JS)
# # ---------------------------------------------------------------
# def download_pdf(pmc_id, title): # Agora 'title' virá correto
    
#     html_page_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/"
    
#     # Limpa o título para usar como nome de arquivo
#     safe_filename = "".join(c for c in title if c.isalnum() or c in (' ', '_')).rstrip()
#     if not safe_filename:
#         safe_filename = pmc_id 
    
#     final_filename = os.path.join(DOWNLOAD_FOLDER, f"{safe_filename[:50]}.pdf")

#     if os.path.exists(final_filename):
#         print(f"[SKIP] Arquivo já existe: {final_filename}")
#         return False
        
#     headers = {
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
#     }

#     # --- FASE 1: ACHAR O LINK DO VISUALIZADOR (Idêntica) ---
#     try:
#         print(f"[HTML] Visitando página do {pmc_id} para achar o botão...")
#         response_html = requests.get(html_page_url, headers=headers, timeout=15)
        
#         if response_html.status_code != 200:
#             print(f"[FAILED] Não consegui visitar a página HTML. Código: {response_html.status_code}\n")
#             return False

#         soup = BeautifulSoup(response_html.text, 'html.parser')
        
#         pdf_link_tag = soup.find('a', attrs={'data-ga-label': 'pdf_download_desktop'})
#         if not pdf_link_tag:
#             pdf_link_tag = soup.find('a', class_='pdf-btn')
        
#         if not pdf_link_tag:
#             print(f"[FAILED] Artigo {pmc_id} é HTML, mas não achei o botão de PDF. Pulando.\n")
#             return False

#         viewer_url = urljoin(html_page_url, pdf_link_tag.get('href'))
#         print(f"[VISUALIZADOR] Botão encontrado! Abrindo visualizador: {viewer_url[:50]}...")

#     except RequestException as e:
#         print(f"[ERROR] Falha na requisição (Fase 1) para {pmc_id}: {e}\n")
#         return False
        
#     # --- FASE 2: ESTRATÉGIA HÍBRIDA (SELENIUM) ---
#     driver = None
#     try:
#         # TENTATIVA 1: Download Automático
#         options_auto = Options()
#         options_auto.add_argument("--headless") 
#         options_auto.add_argument("--disable-gpu")
#         options_auto.add_argument(f"user-agent={headers['User-Agent']}")
#         prefs_auto = {
#             "download.default_directory": os.path.abspath(DOWNLOAD_FOLDER), 
#             "download.prompt_for_download": False,
#             "plugins.always_open_pdf_externally": True, 
#             "plugins.plugins_disabled": ["Chrome PDF Viewer"] 
#         }
#         options_auto.add_experimental_option("prefs", prefs_auto)

#         service = Service(ChromeDriverManager().install())
#         driver = webdriver.Chrome(service=service, options=options_auto)
#         files_before = set(os.listdir(DOWNLOAD_FOLDER))

#         driver.get(viewer_url)
#         print(f"[SELENIUM] Tentativa 1: Download automático...")
#         time.sleep(15) 

#         files_after = set(os.listdir(DOWNLOAD_FOLDER))
#         new_files = files_after - files_before
        
#         # Se falhar, vamos para a TENTATIVA 2 (Clicar)
#         if not new_files:
#             print(f"[SELENIUM] Download automático falhou. Tentativa 2: Clicar no botão...")
#             driver.quit()
            
#             options_click = Options()
#             options_click.add_argument("--headless")
#             options_click.add_argument("--disable-gpu")
#             options_click.add_argument(f"user-agent={headers['User-Agent']}")
#             prefs_click = {
#                 "download.default_directory": os.path.abspath(DOWNLOAD_FOLDER), 
#                 "download.prompt_for_download": False 
#             }
#             options_click.add_experimental_option("prefs", prefs_click)
            
#             driver = webdriver.Chrome(service=service, options=options_click)
#             driver.get(viewer_url)

#             try:
#                 # ===================================================================
#                 # AQUI ESTÁ A CORREÇÃO (Esperar o JS carregar)
#                 # ===================================================================
#                 print(f"[SELENIUM] Esperando JS do visualizador carregar (5s)...")
#                 time.sleep(5) # Espera 5s para o main.js carregar o <pdf-viewer>
                
#                 print(f"[SELENIUM] Esperando o componente '<pdf-viewer>'...")
#                 host_element = WebDriverWait(driver, 30).until(
#                     EC.presence_of_element_located((By.TAG_NAME, "pdf-viewer"))
#                 )
                
#                 print(f"[SELENIUM] Host encontrado! Procurando botão '#download'...")
#                 download_button = driver.execute_script(
#                     "return arguments[0].shadowRoot.querySelector('#download')", 
#                     host_element
#                 )
                
#                 if download_button is None:
#                     raise Exception("Não foi possível achar o botão #download dentro do <pdf-viewer> shadowRoot")

#                 print(f"[SELENIUM] Botão encontrado! Clicando para baixar...")
#                 download_button.click()
#                 time.sleep(15) 
                
#             except Exception as click_error:
#                 print(f"[FAILED] Selenium (Tentativa 2) falhou ao clicar: {click_error}\n")
#                 driver.quit()
#                 return False

#             files_after = set(os.listdir(DOWNLOAD_FOLDER))
#             new_files = files_after - files_before
            
#             if not new_files:
#                 print(f"[FAILED] Selenium clicou, mas nenhum arquivo foi baixado.\n")
#                 driver.quit()
#                 return False

#         # Se chegou aqui, é porque (Tentativa 1 OU 2) funcionou
#         downloaded_filename = new_files.pop()
        
#         if downloaded_filename.endswith(".crdownload"):
#             print("[SELENIUM] Download em progresso... esperando mais 15s...")
#             time.sleep(15)
#             try:
#                 base_name = downloaded_filename.replace(".crdownload", "")
#                 full_path_cr = os.path.join(DOWNLOAD_FOLDER, downloaded_filename)
#                 full_path_base = os.path.join(DOWNLOAD_FOLDER, base_name)
#                 if os.path.exists(full_path_cr):
#                     os.rename(full_path_cr, full_path_base)
#                 downloaded_filename = base_name
#             except Exception as e:
#                 print(f"[FAILED] Não foi possível renomear o arquivo .crdownload: {e}")
#                 driver.quit()
#                 return False
        
#         downloaded_filepath = os.path.join(DOWNLOAD_FOLDER, downloaded_filename)
#         # Renomeia o arquivo baixado (nome genérico) para o nosso nome final (com o TÍTULO)
#         os.rename(downloaded_filepath, final_filename) 
        
#         print(f"[SUCCESS] Salvo e renomeado: {final_filename}\n")
#         driver.quit()
#         return True

#     except Exception as e:
#         print(f"[ERROR] Erro inesperado no Selenium (Fase 2) ao processar {pmc_id}: {e}\n")
#         if driver:
#             driver.quit() 
#         return False

# # ---------------------------------------------------------------
# # FUNÇÃO main (COM O TÍTULO CORRETO E SLEEP DE 5S)
# # ---------------------------------------------------------------

# def main():
#     if YOUR_EMAIL == "seu.email@exemplo.com":
#         print("🚨 ATENÇÃO: Por favor, altere a variável 'YOUR_EMAIL' no script.")
#         return

#     Entrez.email = YOUR_EMAIL
#     setup_environment()
    
#     print("Conectando ao PubMed (via Entrez)...")
    
#     print(f"Buscando '{QUERY}' no PubMed...")
#     handle_search = Entrez.esearch(db="pubmed", term=QUERY, retmax=MAX_ARTICLES)
#     record_search = Entrez.read(handle_search)
#     handle_search.close()
    
#     id_list = record_search["IdList"]
#     if not id_list:
#         print("Nenhum artigo encontrado com essa query.")
#         return

#     print(f"Encontrados {len(id_list)} artigos. Verificando quais têm PDF no PMC...")

#     download_count = 0
#     article_count = 0

#     for pmid in id_list:
#         article_count += 1
        
#         # Aumentamos a pausa para 5s para evitar sermos bloqueados (Erro 403)
#         print(f"\n--- Processando Artigo {article_count}/{len(id_list)} (PMID: {pmid}) ---")
#         time.sleep(5) 
        
#         try:
#             handle_link = Entrez.elink(dbfrom="pubmed", db="pmc", id=pmid, retmode="xml")
#             record_link = Entrez.read(handle_link, validate=False)
#             handle_link.close()
            
#             if not record_link[0]["LinkSetDb"]:
#                 print(f"[INFO] Artigo PMID:{pmid} não está no PMC. Pulando.")
#                 continue

#             pmc_id_num = record_link[0]["LinkSetDb"][0]["Link"][0]["Id"]
#             pmc_id = f"PMC{pmc_id_num}"

#             # ===================================================================
#             # AQUI ESTÁ A CORREÇÃO (PEGANDO O TÍTULO DO PUBMED)
#             # ===================================================================
#             # Usamos o PMID para buscar no 'pubmed' (mais estável)
#             handle_fetch = Entrez.efetch(db="pubmed", id=pmid, rettype="xml", retmode="xml")
#             record_fetch = Entrez.read(handle_fetch, validate=False)
#             handle_fetch.close()
            
#             try:
#                 # Pega o título do XML do PubMed
#                 title = record_fetch['PubmedArticle'][0]['MedlineCitation']['Article']['ArticleTitle']
#                 # Limpa o título (remove colchetes, etc.)
#                 title = str(title).strip().replace('[', '').replace(']', '')
#                 if not title: # Fallback
#                     title = pmc_id
#             except Exception:
#                 title = pmc_id # Fallback
#             # ===================================================================
            
#             if download_pdf(pmc_id, title):
#                 download_count += 1
                
#         except Exception as e:
#             if 'Failed to find tag' in str(e):
#                 print(f"[ERROR] Erro de validação DTD no PMID {pmid}. Pulando. (Detalhe: {e})\n")
#             else:
#                 print(f"[ERROR] Erro inesperado ao processar PMID {pmid}: {e}\n")


#     print("--- FIM DA EXECUÇÃO ---")
#     print(f"Total de artigos analisados: {article_count}")
#     print(f"Total de PDFs baixados: {download_count}")

# if __name__ == "__main__":
#     main()

import os
import time
import requests
from requests.exceptions import RequestException
from Bio import Entrez
from bs4 import BeautifulSoup 
from urllib.parse import urljoin 
import glob 

# --- Bibliotecas do Selenium ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
# (WebDriverWait e EC não são mais necessários, pois não estamos clicando)

# ---------------------------------------------------------------
# CONFIGURAÇÕES
# ---------------------------------------------------------------
QUERY = '((("Radiology"[MeSH Major Topic]) OR (jsubsetr[text]))) AND ("2023/01/01"[Date - Publication] : "2023/12/31"[Date - Publication])'
MAX_ARTICLES = 400 
DOWNLOAD_FOLDER = "Radiology_PDFs"
YOUR_EMAIL = "jadriannassilva@gmail.com" # SEU EMAIL

# ---------------------------------------------------------------
# FUNÇÃO setup_environment (Idêntica)
# ---------------------------------------------------------------
def setup_environment():
    if not os.path.exists(DOWNLOAD_FOLDER):
        os.makedirs(DOWNLOAD_FOLDER)
        print(f"Pasta '{DOWNLOAD_FOLDER}' criada.")

# ---------------------------------------------------------------
# FUNÇÃO download_pdf (v7.0 - Somente Download Automático)
# ---------------------------------------------------------------
def download_pdf(pmc_id, title, title_found): # Recebe 'title' e 'title_found'
    
    html_page_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/"
    
    # ===================================================================
    # NOVA LÓGICA DE NOME (v7.1)
    # ===================================================================
    
    # 1. Se o título não foi encontrado, pulamos, pois você
    #    só quer salvar arquivos com o nome do título.
    if not title_found:
        print(f"[SKIP] Título não encontrado para {pmc_id}. Pulando.")
        return False

    # 2. Gera o nome do arquivo APENAS com o título.
    safe_filename = "".join(c for c in title if c.isalnum() or c in (' ', '_')).rstrip()
    final_filename = os.path.join(DOWNLOAD_FOLDER, f"{safe_filename[:100]}.pdf")

    # 3. Se um arquivo com esse título já existe, pulamos.
    #    Não tentamos renomear com o PMCID.
    if os.path.exists(final_filename):
        print(f"[SKIP] Arquivo com este título já existe: {final_filename}")
        return False
    # ===================================================================
        
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # --- FASE 1: ACHAR O LINK DO VISUALIZADOR ---
    try:
        # CORREÇÃO 403: Pausa ANTES de visitar a página HTML
        print(f"[HTML] Pausando 2s para evitar bloqueio...")
        time.sleep(2) 
        
        print(f"[HTML] Visitando página do {pmc_id} para achar o botão...")
        response_html = requests.get(html_page_url, headers=headers, timeout=15)
        
        if response_html.status_code != 200:
            print(f"[FAILED] Não consegui visitar a página HTML. Código: {response_html.status_code}\n")
            return False

        soup = BeautifulSoup(response_html.text, 'html.parser')
        
        pdf_link_tag = soup.find('a', attrs={'data-ga-label': 'pdf_download_desktop'})
        if not pdf_link_tag:
            pdf_link_tag = soup.find('a', class_='pdf-btn')
        
        if not pdf_link_tag:
            print(f"[FAILED] Artigo {pmc_id} é HTML, mas não achei o botão de PDF. Pulando.\n")
            return False

        viewer_url = urljoin(html_page_url, pdf_link_tag.get('href'))
        print(f"[VISUALIZADOR] Botão encontrado! Abrindo visualizador: {viewer_url[:50]}...")

    except RequestException as e:
        print(f"[ERROR] Falha na requisição (Fase 1) para {pmc_id}: {e}\n")
        return False
        
    # --- FASE 2: TENTATIVA DE DOWNLOAD AUTOMÁTICO (SELENIUM) ---
    driver = None
    try:
        options_auto = Options()
        options_auto.add_argument("--headless") 
        options_auto.add_argument("--disable-gpu")
        options_auto.add_argument(f"user-agent={headers['User-Agent']}")
        prefs_auto = {
            "download.default_directory": os.path.abspath(DOWNLOAD_FOLDER), 
            "download.prompt_for_download": False,
            "plugins.always_open_pdf_externally": True, 
            "plugins.plugins_disabled": ["Chrome PDF Viewer"] 
        }
        options_auto.add_experimental_option("prefs", prefs_auto)

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options_auto)
        files_before = set(os.listdir(DOWNLOAD_FOLDER))

        driver.get(viewer_url)
        print(f"[SELENIUM] Download automático... (esperando 15s)")
        time.sleep(15) 

        files_after = set(os.listdir(DOWNLOAD_FOLDER))
        new_files = files_after - files_before
        
        if not new_files:
            print(f"[FAILED] Chrome visitou a URL, mas nenhum arquivo foi baixado (provavelmente é um visualizador HTML).\n")
            driver.quit()
            return False

        # Se chegou aqui, o download automático funcionou
        downloaded_filename = new_files.pop()
        
        if downloaded_filename.endswith(".crdownload"):
            print("[SELENIUM] Download em progresso... esperando mais 15s...")
            time.sleep(15)
            try:
                base_name = downloaded_filename.replace(".crdownload", "")
                full_path_cr = os.path.join(DOWNLOAD_FOLDER, downloaded_filename)
                full_path_base = os.path.join(DOWNLOAD_FOLDER, base_name)
                if os.path.exists(full_path_cr):
                    os.rename(full_path_cr, full_path_base)
                downloaded_filename = base_name
            except Exception as e:
                print(f"[FAILED] Não foi possível renomear o arquivo .crdownload: {e}")
                driver.quit()
                return False
        
        downloaded_filepath = os.path.join(DOWNLOAD_FOLDER, downloaded_filename)
        os.rename(downloaded_filepath, final_filename) # Renomeia para o NOME FINAL (só o título)
        
        print(f"[SUCCESS] Salvo e renomeado: {final_filename}\n")
        driver.quit()
        return True

    except Exception as e:
        print(f"[ERROR] Erro inesperado no Selenium (Fase 2) ao processar {pmc_id}: {e}\n")
        if driver:
            driver.quit() 
        return False

# ---------------------------------------------------------------
# FUNÇÃO main (COM TÍTULO E SLEEP DE 10S)
# ---------------------------------------------------------------

def main():
    if YOUR_EMAIL == "seu.email@exemplo.com":
        print("🚨 ATENÇÃO: Por favor, altere a variável 'YOUR_EMAIL' no script.")
        return

    Entrez.email = YOUR_EMAIL
    setup_environment()
    
    print("Conectando ao PubMed (via Entrez)...")
    
    print(f"Buscando '{QUERY}' no PubMed...")
    handle_search = Entrez.esearch(db="pubmed", term=QUERY, retmax=MAX_ARTICLES)
    record_search = Entrez.read(handle_search)
    handle_search.close()
    
    id_list = record_search["IdList"]
    if not id_list:
        print("Nenhum artigo encontrado com essa query.")
        return

    print(f"Encontrados {len(id_list)} artigos. Verificando quais têm PDF no PMC...")

    download_count = 0
    article_count = 0

    for pmid in id_list:
        article_count += 1
        
        # Aumentamos a pausa para 10s para evitar sermos bloqueados (Erro 403)
        print(f"\n--- Processando Artigo {article_count}/{len(id_list)} (PMID: {pmid}) ---")
        time.sleep(10) 
        
        try:
            handle_link = Entrez.elink(dbfrom="pubmed", db="pmc", id=pmid, retmode="xml")
            record_link = Entrez.read(handle_link, validate=False)
            handle_link.close()
            
            if not record_link[0]["LinkSetDb"]:
                print(f"[INFO] Artigo PMID:{pmid} não está no PMC. Pulando.")
                continue

            pmc_id_num = record_link[0]["LinkSetDb"][0]["Link"][0]["Id"]
            pmc_id = f"PMC{pmc_id_num}"

            handle_fetch = Entrez.efetch(db="pubmed", id=pmid, rettype="xml", retmode="xml")
            record_fetch = Entrez.read(handle_fetch, validate=False)
            handle_fetch.close()
            
            title_found = False
            title = pmc_id 
            try:
                article_title = record_fetch['PubmedArticle'][0]['MedlineCitation']['Article']['ArticleTitle']
                article_title = str(article_title).strip().replace('[', '').replace(']', '')
                if article_title:
                    title = article_title
                    title_found = True
            except Exception as e:
                print(f"[AVISO] Falha ao extrair título do PMID {pmid}. Usando ID. Erro: {e}")
            
            if download_pdf(pmc_id, title, title_found): # Passa o título e o flag
                download_count += 1
                
        except Exception as e:
            if 'Failed to find tag' in str(e):
                print(f"[ERROR] Erro de validação DTD no PMID {pmid}. Pulando. (Detalhe: {e})\n")
            else:
                print(f"[ERROR] Erro inesperado ao processar PMID {pmid}: {e}\n")


    print("--- FIM DA EXECUÇÃO ---")
    print(f"Total de artigos analisados: {article_count}")
    print(f"Total de PDFs baixados: {download_count}")

if __name__ == "__main__":
    main()