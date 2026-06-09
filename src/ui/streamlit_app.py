"""
streamlit_app.py — Interface de Chat para o Podcast Q&A Assistant.

Funcionalidades:
- Chat com histórico de mensagens
- Busca RAG nas transcrições de vídeos
- Function-calling com tool get_timestamp
- Cache semântico (mostra hit/miss)
- Métricas na sidebar (custo, latência, cache hit-rate)

Rodar:
    streamlit run src/ui/streamlit_app.py
"""

import os
import sys
import time
import json

import streamlit as st
from dotenv import load_dotenv

# Adicionar o diretório raiz ao path para imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

load_dotenv()

from src.pipeline.rag import generate_answer, generate_answer_stream, get_chroma_collection, retrieve
from src.pipeline.tools import call_with_tools, get_timestamp, list_videos
from src.pipeline.cache import SemanticCache
from src.pipeline.routing import ModelRouter
from src.observability.trace import log_query, timed

# ============================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================
st.set_page_config(
    page_title="🎙️ Sintetiza",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================
# CSS CUSTOMIZADO
# ============================================
st.markdown("""
<style>
    /* Tema escuro moderno */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Header */
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        color: #8892b0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Chat messages */
    .stChatMessage {
        border-radius: 12px !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Source cards */
    .source-card {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 10px;
        padding: 12px 16px;
        margin: 6px 0;
        transition: all 0.2s ease;
    }
    
    .source-card:hover {
        background: rgba(102, 126, 234, 0.2);
        border-color: rgba(102, 126, 234, 0.5);
        transform: translateY(-1px);
    }
    
    .source-card a {
        color: #667eea !important;
        text-decoration: none;
        font-weight: 600;
    }
    
    .source-card a:hover {
        color: #764ba2 !important;
    }
    
    /* Metrics */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 16px;
        margin: 8px 0;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #8892b0;
        margin-top: 4px;
    }
    
    /* Cache badge */
    .cache-hit {
        background: rgba(0, 200, 83, 0.15);
        color: #00c853;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .cache-miss {
        background: rgba(255, 152, 0, 0.15);
        color: #ff9800;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Tool badge */
    .tool-badge {
        background: rgba(156, 39, 176, 0.15);
        color: #ce93d8;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-left: 8px;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(15, 15, 26, 0.95) !important;
    }
    
    /* Loading animation */
    .loading-dots {
        display: inline-block;
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 1; }
    }
    
    /* Divider */
    .gradient-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent);
        border: none;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# INICIALIZAÇÃO DO STATE
# ============================================

def init_state():
    """Inicializa o session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "cache" not in st.session_state:
        st.session_state.cache = SemanticCache()
    if "router" not in st.session_state:
        st.session_state.router = ModelRouter()
    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0
    if "query_count" not in st.session_state:
        st.session_state.query_count = 0
    if "latencies" not in st.session_state:
        st.session_state.latencies = []

init_state()


# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.markdown("## ⚙️ Configuração")
    
    # Verificar API Key
    api_key = os.getenv("GROQ_API_KEY", "")
    if api_key and api_key != "gsk_sua_chave_aqui":
        st.success("✅ API Key configurada")
    else:
        st.error("❌ Configure GROQ_API_KEY no .env")
        api_key_input = st.text_input("Cole sua API Key:", type="password")
        if api_key_input:
            os.environ["GROQ_API_KEY"] = api_key_input
            st.rerun()
    
    # Verificar corpus
    try:
        collection = get_chroma_collection()
        doc_count = collection.count()
        if doc_count > 0:
            st.success(f"📚 {doc_count} chunks indexados")
        else:
            st.warning("⚠️ Nenhum dado indexado. Rode o setup primeiro.")
    except Exception:
        doc_count = 0
        st.warning("⚠️ ChromaDB não disponível")
    
    st.markdown('')
    
    # Métricas
    st.markdown("## 📊 Métricas")
    
    cache_stats = st.session_state.cache.get_stats()
    router_stats = st.session_state.router.get_stats()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", st.session_state.query_count)
        st.metric("Cache Hits", cache_stats["cache_hits"])
    with col2:
        st.metric("Hit Rate", cache_stats["hit_rate"])
        st.metric("Tokens Salvos", cache_stats["tokens_saved"])
    
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    # Routing stats
    st.markdown("### 🔀 Model Routing")
    if router_stats["total_requests"] > 0:
        st.write(f"🟢 Modelo barato: **{router_stats['cheap_model_calls']}x** ({router_stats['cheap_ratio']})")
        st.write(f"🔵 Modelo forte: **{router_stats['strong_model_calls']}x**")
        st.write(f"💰 Economia: **{router_stats['savings_percent']}**")
    else:
        st.write("_Nenhuma query ainda_")
    
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    # Latência
    st.markdown("### ⏱️ Latência")
    if st.session_state.latencies:
        latencies = st.session_state.latencies
        avg_lat = sum(latencies) / len(latencies)
        p95_lat = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 2 else latencies[-1]
        st.write(f"Média: **{avg_lat:.2f}s**")
        st.write(f"P95: **{p95_lat:.2f}s**")
    else:
        st.write("_Nenhuma query ainda_")
    
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    # Vídeos disponíveis
    st.markdown("### 🎬 Vídeos no Corpus")
    try:
        videos = list_videos()
        for v in videos.get("videos", []):
            st.write(f"📺 [{v['title']}]({v['url']})")
    except Exception:
        st.write("_Nenhum vídeo indexado_")
    
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    # Botões de ação
    if st.button("🗑️ Limpar Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("🔄 Limpar Cache", use_container_width=True):
        st.session_state.cache.clear()
        st.rerun()


# ============================================
# ÁREA PRINCIPAL — CHAT
# ============================================

# Header
st.markdown('<div class="main-header">🎙️ Sintetiza</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Faça perguntas sobre seus podcasts e vídeos favoritos e '
    'receba respostas com link direto para o trecho exato!'
    '</div>',
    unsafe_allow_html=True,
)

# Mensagem de boas-vindas
if not st.session_state.messages:
    st.markdown("""
    > 👋 **Bem-vindo!** Eu sou seu assistente de busca em podcasts e vídeos.
    > 
    > **O que posso fazer:**
    > - 🔍 Responder perguntas sobre o conteúdo dos vídeos
    > - ⏱️ Encontrar o trecho exato com timestamp e link
    > - 📋 Listar os vídeos disponíveis
    > 
    > **Experimente perguntar:**
    > - _"O que é explicado sobre machine learning?"_
    > - _"Em qual momento do vídeo se fala sobre API?"_
    > - _"Quais vídeos estão disponíveis?"_
    """)

# Exibir mensagens anteriores
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Mostrar badges
        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            badges = ""
            if meta.get("cache_hit"):
                badges += '<span class="cache-hit">⚡ CACHE HIT</span>'
            else:
                badges += '<span class="cache-miss">🌐 API CALL</span>'
            if meta.get("tool_used"):
                badges += f'<span class="tool-badge">🔧 {meta["tool_used"]}</span>'
            
            st.markdown(badges, unsafe_allow_html=True)
            
            # Mostrar sources
            if meta.get("sources"):
                with st.expander("📚 Fontes utilizadas"):
                    for src in meta["sources"][:3]:
                        st.markdown(
                            f'<div class="source-card">'
                            f'📺 <b>{src["video_title"]}</b> ⏱️ {src["timestamp"]}<br>'
                            f'🔗 <a href="{src["url"]}" target="_blank">Assistir trecho</a>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )


# ============================================
# INPUT DO USUÁRIO
# ============================================

if prompt := st.chat_input("Faça sua pergunta sobre os podcasts/vídeos..."):
    
    # Adicionar mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Processar resposta
    with st.chat_message("assistant"):
        start_time = time.time()
        
        with st.spinner("🔍 Buscando nos vídeos..."):
            
            # 1. Verificar cache
            cached = st.session_state.cache.lookup(prompt)
            
            if cached:
                # Cache hit!
                answer = cached["answer"]
                model = "cache"
                tokens = 0
                sources = cached.get("sources", [])
                tool_used = None
                cache_hit = True
                cost = 0.0
            else:
                # 2. Model routing
                model = st.session_state.router.route(prompt)
                
                # 3. Decidir: usar tools ou RAG direto?
                # Se a pergunta pede timestamp/link ou lista → usa tools
                use_tools = any(kw in prompt.lower() for kw in [
                    "timestamp", "momento", "minuto", "trecho",
                    "link", "quando", "quais vídeos", "listar",
                    "onde ele fala", "em que parte",
                ])
                
                if use_tools:
                    # Function-calling com tools
                    result = call_with_tools(prompt, model=model)
                    tool_used = result["tool_calls"][0]["tool"] if result["tool_calls"] else None
                    answer = result["answer"]
                    tokens = result.get("tokens_used", 0)
                    sources = result.get("sources", [])
                    st.markdown(answer)
                else:
                    # RAG direto com Streaming
                    meta_dict = {}
                    answer = st.write_stream(generate_answer_stream(prompt, meta_dict, model=model))
                    tool_used = None
                    tokens = meta_dict.get("tokens_used", 0)
                    sources = meta_dict.get("sources", [])
                
                cache_hit = False
                cost = st.session_state.router.record_cost(model, tokens)
                
                # 4. Salvar no cache
                st.session_state.cache.store(prompt, {
                    "answer": answer,
                    "model": model,
                    "tokens_used": tokens,
                    "sources": sources,
                })
        
        elapsed = time.time() - start_time
        
        # Exibir resposta para cache_hit (para streaming já foi exibido)
        if cache_hit:
            st.markdown(answer)
        
        # Badges
        badges = ""
        if cache_hit:
            badges += f'<span class="cache-hit">⚡ CACHE HIT (TTL: {st.session_state.cache.get_stats()["ttl_seconds"]//3600}h)</span>'
        else:
            badges += '<span class="cache-miss">🌐 API CALL</span>'
        if tool_used:
            badges += f'<span class="tool-badge">🔧 {tool_used}</span>'
        
        st.markdown(badges, unsafe_allow_html=True)
        
        # Fontes
        if sources and not cache_hit:
            with st.expander("📚 Fontes utilizadas"):
                for src in sources[:3]:
                    st.markdown(
                        f'<div class="source-card">'
                        f'📺 <b>{src["video_title"]}</b> ⏱️ {src["timestamp"]}<br>'
                        f'🔗 <a href="{src["url"]}" target="_blank">Assistir trecho</a>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        
        # Info de performance (pequeno)
        perf_text = f"⏱️ {elapsed:.2f}s"
        if not cache_hit:
            perf_text += f" | 🤖 {model} | 📊 {tokens} tokens"
        st.caption(perf_text)
        
        # Salvar mensagem no histórico
        msg_meta = {
            "cache_hit": cache_hit,
            "tool_used": tool_used,
            "model": model,
            "tokens": tokens,
            "latency": elapsed,
            "sources": sources,
        }
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "meta": msg_meta,
        })
        
        # Atualizar métricas globais
        st.session_state.query_count += 1
        st.session_state.total_tokens += tokens
        st.session_state.total_cost += cost
        st.session_state.latencies.append(elapsed)
        
        # Log
        log_query(
            query=prompt,
            model=model,
            tokens=tokens,
            latency=elapsed,
            cache_hit=cache_hit,
            tool_used=tool_used,
            cost=cost,
        )
