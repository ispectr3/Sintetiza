"""
rag.py — Pipeline RAG completo: chunking, embeddings e retrieval.

Responsável por:
1. Ler as transcrições JSON
2. Dividir em chunks com metadata de timestamp
3. Gerar embeddings e armazenar no ChromaDB
4. Buscar chunks relevantes para uma pergunta
5. Gerar resposta usando Groq (Llama 3.3)
"""

import json
import os
from pathlib import Path

import chromadb
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ============================================
# CONFIGURAÇÃO
# ============================================
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
CHROMA_DB_PATH = os.path.join(_PROJECT_ROOT, "chroma_db")
COLLECTION_NAME = "podcast_transcripts"
CHUNK_SIZE = 800        # caracteres por chunk
CHUNK_OVERLAP = 100     # overlap entre chunks
TOP_K = 5               # quantos chunks retornar na busca


def get_groq_client() -> Groq:
    """Retorna cliente Groq configurado."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "❌ GROQ_API_KEY não encontrada!\n"
            "   Crie sua chave grátis em https://console.groq.com\n"
            "   Depois adicione no .env: GROQ_API_KEY=gsk_sua_chave"
        )
    return Groq(api_key=api_key)


def get_chroma_collection():
    """Retorna a collection do ChromaDB (cria se não existir)."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


# ============================================
# CHUNKING — Dividir transcrições em pedaços
# ============================================

def chunk_transcript(transcript_data: dict) -> list[dict]:
    """
    Divide uma transcrição em chunks com metadata de timestamp.
    
    Cada chunk contém:
    - text: o texto do chunk
    - video_id: ID do vídeo no YouTube
    - video_title: título do vídeo
    - video_url: URL completa do vídeo
    - start_time: timestamp de início (segundos)
    - end_time: timestamp de fim (segundos)
    
    Args:
        transcript_data: dados da transcrição (do ingest.py)
    
    Returns:
        Lista de chunks com metadata
    """
    segments = transcript_data["segments"]
    video_id = transcript_data["video_id"]
    video_title = transcript_data["video_title"]
    video_url = transcript_data["video_url"]
    
    chunks = []
    current_text = ""
    current_start = segments[0]["start"] if segments else 0
    current_segments = []
    
    for segment in segments:
        segment_text = segment["text"].strip()
        
        # Se adicionar este segmento ultrapassa o tamanho do chunk
        if len(current_text) + len(segment_text) + 1 > CHUNK_SIZE and current_text:
            # Salva o chunk atual
            end_time = current_segments[-1]["start"] + current_segments[-1]["duration"]
            chunks.append({
                "text": current_text.strip(),
                "video_id": video_id,
                "video_title": video_title,
                "video_url": video_url,
                "start_time": current_start,
                "end_time": end_time,
            })
            
            # Overlap: manter os últimos segmentos que somam ~CHUNK_OVERLAP caracteres
            overlap_text = ""
            overlap_segments = []
            for seg in reversed(current_segments):
                if len(overlap_text) + len(seg["text"]) > CHUNK_OVERLAP:
                    break
                overlap_text = seg["text"] + " " + overlap_text
                overlap_segments.insert(0, seg)
            
            current_text = overlap_text.strip() + " " if overlap_text else ""
            current_segments = overlap_segments.copy()
            current_start = overlap_segments[0]["start"] if overlap_segments else segment["start"]
        
        current_text += segment_text + " "
        current_segments.append(segment)
    
    # Último chunk
    if current_text.strip():
        end_time = current_segments[-1]["start"] + current_segments[-1]["duration"] if current_segments else current_start
        chunks.append({
            "text": current_text.strip(),
            "video_id": video_id,
            "video_title": video_title,
            "video_url": video_url,
            "start_time": current_start,
            "end_time": end_time,
        })
    
    return chunks


def load_and_chunk_all(corpus_dir: str = "data/corpus/transcripts") -> list[dict]:
    """
    Carrega todas as transcrições e divide em chunks.
    
    Returns:
        Lista de todos os chunks de todos os vídeos
    """
    corpus_path = Path(corpus_dir)
    all_chunks = []
    
    if not corpus_path.exists():
        print(f"❌ Pasta {corpus_dir} não encontrada!")
        print("   Rode primeiro: python -m src.pipeline.ingest")
        return []
    
    json_files = list(corpus_path.glob("*.json"))
    
    if not json_files:
        print(f"❌ Nenhuma transcrição encontrada em {corpus_dir}/")
        print("   Rode primeiro: python -m src.pipeline.ingest")
        return []
    
    print(f"📄 {len(json_files)} transcrições encontradas")
    
    for filepath in json_files:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        chunks = chunk_transcript(data)
        all_chunks.extend(chunks)
        print(f"   📎 {data['video_title']}: {len(chunks)} chunks")
    
    print(f"\n📊 Total: {len(all_chunks)} chunks criados")
    return all_chunks


# ============================================
# EMBEDDINGS — Armazenar no ChromaDB
# ============================================

def index_chunks(chunks: list[dict]) -> int:
    """
    Gera embeddings e armazena os chunks no ChromaDB.
    
    ChromaDB usa o modelo default (all-MiniLM-L6-v2) para
    gerar embeddings automaticamente.
    
    Returns:
        Número de chunks indexados
    """
    if not chunks:
        print("❌ Nenhum chunk para indexar!")
        return 0
    
    collection = get_chroma_collection()
    
    # Verificar se já tem dados
    existing = collection.count()
    if existing > 0:
        print(f"⚠️  Collection já tem {existing} chunks. Limpando...")
        # Deletar collection existente e recriar
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        client.delete_collection(COLLECTION_NAME)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    
    print(f"🔄 Indexando {len(chunks)} chunks no ChromaDB...")
    
    # Preparar dados para inserção
    ids = []
    documents = []
    metadatas = []
    
    for i, chunk in enumerate(chunks):
        ids.append(f"chunk_{i}")
        documents.append(chunk["text"])
        metadatas.append({
            "video_id": chunk["video_id"],
            "video_title": chunk["video_title"],
            "video_url": chunk["video_url"],
            "start_time": chunk["start_time"],
            "end_time": chunk["end_time"],
            "timestamp_url": f"{chunk['video_url']}&t={int(chunk['start_time'])}s",
        })
    
    # Inserir em batches de 100 (limite do Chroma)
    batch_size = 100
    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"   ✅ Batch {start // batch_size + 1}: chunks {start}–{end - 1}")
    
    print(f"\n🎉 {len(ids)} chunks indexados com sucesso!")
    return len(ids)


# ============================================
# RETRIEVAL — Buscar chunks relevantes
# ============================================

def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Busca os chunks mais relevantes para uma pergunta.
    
    Args:
        query: pergunta do usuário
        top_k: quantos chunks retornar
    
    Returns:
        Lista de dicts com text, metadata e distance
    """
    collection = get_chroma_collection()
    
    if collection.count() == 0:
        print("❌ Nenhum dado no ChromaDB! Rode o indexador primeiro.")
        return []
    
    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
    )
    
    # Formatar resultados
    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i] if results["distances"] else None,
        })
    
    return retrieved


# ============================================
# GENERATION — Gerar resposta com Groq
# ============================================

SYSTEM_PROMPT = """Você é um assistente especializado em responder perguntas sobre podcasts e vídeos do YouTube.

REGRAS:
1. Responda APENAS com base no contexto fornecido (trechos de transcrições).
2. Se a resposta não estiver no contexto, diga claramente "Não encontrei essa informação nas transcrições disponíveis."
3. Sempre cite o vídeo e o timestamp quando possível.
4. Responda em português do Brasil.
5. Seja conciso e direto.
6. Quando mencionar um trecho, inclua o link com timestamp para o usuário acessar.

FORMATO DA RESPOSTA:
- Responda a pergunta de forma clara
- Cite as fontes: "📺 [Título do Vídeo] — [timestamp]"
- Se tiver link, mostre: "🔗 [URL com timestamp]"
"""

def build_context(retrieved_chunks: list[dict]) -> str:
    """Monta o contexto a partir dos chunks recuperados."""
    context_parts = []
    
    for i, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk["metadata"]
        minutes = int(meta["start_time"]) // 60
        seconds = int(meta["start_time"]) % 60
        timestamp = f"{minutes:02d}:{seconds:02d}"
        
        context_parts.append(
            f"--- Trecho {i} ---\n"
            f"Vídeo: {meta['video_title']}\n"
            f"Timestamp: {timestamp}\n"
            f"Link: {meta['timestamp_url']}\n"
            f"Texto: {chunk['text']}\n"
        )
    
    return "\n".join(context_parts)


def generate_answer(
    query: str,
    model: str = None,
    top_k: int = TOP_K,
) -> dict:
    """
    Pipeline RAG completo: retrieve → build context → generate.
    
    Args:
        query: pergunta do usuário
        model: modelo Groq (default: GROQ_MODEL do .env)
        top_k: quantos chunks usar como contexto
    
    Returns:
        Dict com: answer, sources, model, tokens_used
    """
    if model is None:
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    # 1. Retrieve
    retrieved = retrieve(query, top_k)
    
    if not retrieved:
        return {
            "answer": "❌ Não há dados indexados. Rode o indexador primeiro.",
            "sources": [],
            "model": model,
            "tokens_used": 0,
        }
    
    # 2. Build context
    context = build_context(retrieved)
    
    # 3. Generate
    client = get_groq_client()
    
    user_message = (
        f"CONTEXTO (trechos de transcrições de vídeos):\n\n"
        f"{context}\n\n"
        f"PERGUNTA DO USUÁRIO:\n{query}"
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    
    answer = response.choices[0].message.content
    tokens_used = response.usage.total_tokens if response.usage else 0
    
    # Formatar sources
    sources = []
    for chunk in retrieved:
        meta = chunk["metadata"]
        minutes = int(meta["start_time"]) // 60
        seconds = int(meta["start_time"]) % 60
        sources.append({
            "video_title": meta["video_title"],
            "timestamp": f"{minutes:02d}:{seconds:02d}",
            "url": meta["timestamp_url"],
            "relevance": 1 - chunk["distance"] if chunk["distance"] else None,
        })
    
    return {
        "answer": answer,
        "sources": sources,
        "model": model,
        "tokens_used": tokens_used,
    }


def generate_answer_stream(
    query: str,
    meta_dict: dict,
    model: str = None,
    top_k: int = TOP_K,
):
    """
    Pipeline RAG com suporte a streaming via yield.
    O meta_dict é passado por referência para ser preenchido durante/após a stream.
    """
    if model is None:
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    meta_dict["model"] = model
    retrieved = retrieve(query, top_k)
    
    if not retrieved:
        meta_dict["sources"] = []
        meta_dict["tokens_used"] = 0
        meta_dict["answer"] = "❌ Não há dados indexados. Rode o indexador primeiro."
        yield meta_dict["answer"]
        return
        
    context = build_context(retrieved)
    
    sources = []
    for chunk in retrieved:
        meta = chunk["metadata"]
        minutes = int(meta["start_time"]) // 60
        seconds = int(meta["start_time"]) % 60
        sources.append({
            "video_title": meta["video_title"],
            "timestamp": f"{minutes:02d}:{seconds:02d}",
            "url": meta["timestamp_url"],
            "relevance": 1 - chunk["distance"] if chunk["distance"] else None,
        })
    meta_dict["sources"] = sources
    
    client = get_groq_client()
    user_message = (
        f"CONTEXTO (trechos de transcrições de vídeos):\n\n"
        f"{context}\n\n"
        f"PERGUNTA DO USUÁRIO:\n{query}"
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=1024,
        stream=True
    )
    
    full_answer = ""
    tokens = 0
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            full_answer += content
            yield content
        if hasattr(chunk, 'x_groq') and chunk.x_groq and hasattr(chunk.x_groq, 'usage') and chunk.x_groq.usage:
            tokens = chunk.x_groq.usage.total_tokens
            
    # Fallback tokens if not returned in stream chunks
    if tokens == 0:
        tokens = int(len(full_answer) / 4) + int(len(context) / 4)
        
    meta_dict["tokens_used"] = tokens
    meta_dict["answer"] = full_answer


# ============================================
# SETUP — Rodar indexação completa
# ============================================

def setup_rag():
    """Roda o pipeline completo de setup: load → chunk → index."""
    print("\n🚀 SETUP DO PIPELINE RAG\n")
    
    # 1. Carregar e dividir em chunks
    chunks = load_and_chunk_all()
    
    if not chunks:
        return False
    
    # 2. Indexar no ChromaDB
    count = index_chunks(chunks)
    
    if count > 0:
        print(f"\n✅ Pipeline RAG pronto! {count} chunks indexados.")
        print("   Agora rode: streamlit run src/ui/streamlit_app.py")
        return True
    
    return False


if __name__ == "__main__":
    setup_rag()
