const $ = (sel) => document.querySelector(sel);
const chat  = $('#chat');
const input = $('#input');
const sendBtn = $('#send');
const AVATAR_SRC = 'mascote.png';

// API: prod vs dev
const PROD_API = "https://chat-deulaudo.onrender.com/ask";
const DEV_API  = "http://127.0.0.1:8000/ask";           
const API_URL  = (location.hostname === "localhost" || location.hostname === "127.0.0.1")
  ? DEV_API : PROD_API;

const VECTOR_STORE_NAME = "rag100";

function toCleanHtml(s) {
  if (!s) return "";
  let out = String(s);
  out = out.replace(/```[\s\S]*?```/g, "");
  out = out.replace(/^#{1,6}\s+/gm, "");
  out = out.replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>");
  out = out.replace(/(^|[^\*])\*([^*\n]+)\*(?!\*)/g, "$1$2");
  out = out.replace(/(^|[^_])_([^_\n]+)_(?!_)/g, "$1$2");
  out = out.replace(/^\s*[-*]\s+/gm, "");
  out = out.replace(/^(?!\d+\.)\s*([^\n:]{3,}):\s*$/gm, (m, g1) => `<b>${g1.toUpperCase()}:</b>`);
  out = out.replace(/\r/g, "").replace(/\n{3,}/g, "\n\n").replace(/\n/g, "<br>");
  return out.trim();
}

// UI helpers (iguais aos seus)
function addMessage(role, htmlOrText, meta) {
  const wrap = document.createElement('div');
  wrap.className = `msg ${role}`;

  if (role === 'bot') {
    const img = document.createElement('img');
    img.src = AVATAR_SRC; img.alt = 'Mascote'; img.className = 'avatar';
    wrap.appendChild(img);
  }
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = htmlOrText || '';
  // se vier texto puro, converta:
  if (!htmlOrText?.includes('<b') && !htmlOrText?.includes('<br')) {
    bubble.innerHTML = toCleanHtml(htmlOrText);
  }
  wrap.appendChild(bubble);
  chat.appendChild(wrap);

  if (meta) { const m = document.createElement('div'); m.className='meta'; m.textContent=meta; chat.appendChild(m); }
  chat.scrollTop = chat.scrollHeight; wrap.classList.add('in');
}

function addUserMessage(text) {
  const wrap = document.createElement('div');
  wrap.className = 'msg user';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  wrap.appendChild(bubble);
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
  wrap.classList.add('in');
}

function setLoading(on){ sendBtn.disabled = !!on; sendBtn.textContent = on ? 'Enviando…' : 'Enviar'; }
function autoresizeTextarea(el){ el.style.height='auto'; el.style.height=Math.min(el.scrollHeight,160)+'px'; }

async function send() {
  const pergunta = input.value.trim();
  if (!pergunta) return;
  input.value=''; autoresizeTextarea(input);
  addUserMessage(pergunta);

  const thinking = document.createElement('div');
  thinking.className='meta';
  thinking.innerHTML='<span class="typing"></span> Respondendo…';
  chat.appendChild(thinking); chat.scrollTop = chat.scrollHeight;

  setLoading(true);
  const t0 = performance.now();
  
  try {
    // 2. Fazer o fetch com método POST, mas SEM body
    const res = await fetch(API_URL, {
      method: 'POST',
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        q: pergunta,
        vs_name: VECTOR_STORE_NAME
      })
    });
    
    // --- FIM DA MUDANÇA ---

    const dt = Math.round(performance.now() - t0);
    const data = await res.json().catch(()=> ({}));
    thinking.remove();

    // 3. Ajustar a leitura da resposta
    if (!res.ok || data.error) {
      addMessage('bot', `Erro: ${data.error || data.detail || res.status}`);
    } else {
      const txt = data.response || '';
      addMessage('bot', txt, `latência: ${dt} ms`);
    }
  } catch (err) {
    thinking.remove();
    addMessage('bot', `Falha de rede: ${err}`);
  } finally {
    setLoading(false); input.focus();
  }
}

sendBtn.addEventListener('click', send);
input.addEventListener('keydown', (e)=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); send(); }});
input.addEventListener('input', ()=> autoresizeTextarea(input));
setTimeout(()=>{ input.focus(); autoresizeTextarea(input); }, 100);


