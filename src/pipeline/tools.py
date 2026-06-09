"""
tools.py — Tools customizadas para function-calling.

Tool principal: get_timestamp
- Busca o trecho exato de um podcast/vídeo
- Retorna timestamp formatado (mm:ss) e URL com link direto

Esta tool é usada pelo LLM via function-calling para
dar respostas mais precisas com links para o trecho exato.
"""

import json
import os
from src.pipeline.rag import retrieve, get_groq_client


# ============================================
# TOOL 1: get_timestamp (tool do domínio)
# ============================================

def get_timestamp(quote: str) -> dict:
    """
    Busca o timestamp e URL do trecho onde uma frase/tópico aparece no vídeo.
    
    Args:
        quote: frase ou tópico a buscar nos vídeos
    
    Returns:
        Dict com: timestamp (mm:ss), url (link direto), video_title, text
    """
    # Buscar o trecho mais similar
    results = retrieve(quote, top_k=1)
    
    if not results:
        return {
            "found": False,
            "message": "Nenhum trecho encontrado para essa busca.",
        }
    
    best_match = results[0]
    meta = best_match["metadata"]
    
    # Formatar timestamp
    start_seconds = int(float(meta["start_time"]))
    minutes = start_seconds // 60
    seconds = start_seconds % 60
    
    return {
        "found": True,
        "timestamp": f"{minutes:02d}:{seconds:02d}",
        "url": meta["timestamp_url"],
        "video_title": meta["video_title"],
        "video_id": meta["video_id"],
        "text": best_match["text"][:300],  # preview do texto
        "relevance_score": round(1 - best_match["distance"], 3) if best_match["distance"] else None,
    }


# ============================================
# TOOL 2: list_videos (tool auxiliar)
# ============================================

def list_videos() -> dict:
    """
    Lista todos os vídeos disponíveis no corpus.
    
    Returns:
        Dict com lista de vídeos {title, video_id, url}
    """
    from src.pipeline.rag import get_chroma_collection
    
    collection = get_chroma_collection()
    
    if collection.count() == 0:
        return {"videos": [], "message": "Nenhum vídeo indexado."}
    
    # Pegar todos os metadados únicos
    all_data = collection.get(include=["metadatas"])
    
    videos_seen = set()
    videos = []
    
    for meta in all_data["metadatas"]:
        vid = meta["video_id"]
        if vid not in videos_seen:
            videos_seen.add(vid)
            videos.append({
                "video_id": vid,
                "title": meta["video_title"],
                "url": meta["video_url"],
            })
    
    return {
        "videos": videos,
        "total": len(videos),
    }


# ============================================
# DEFINIÇÃO DAS TOOLS PARA FUNCTION-CALLING
# ============================================

# Definição no formato OpenAI/Groq para function-calling
TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "get_timestamp",
            "description": (
                "Busca o timestamp exato e URL de um trecho específico "
                "nos vídeos/podcasts indexados. Use quando o usuário quer "
                "encontrar onde algo específico foi dito, ou quer o link "
                "direto para um trecho do vídeo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "quote": {
                        "type": "string",
                        "description": (
                            "A frase, tópico ou assunto a buscar nos vídeos. "
                            "Exemplo: 'explicação sobre clean code' ou "
                            "'definição de machine learning'"
                        ),
                    },
                },
                "required": ["quote"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_videos",
            "description": (
                "Lista todos os vídeos/podcasts disponíveis no corpus. "
                "Use quando o usuário pergunta quais vídeos estão "
                "disponíveis ou quer ver o catálogo."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

# Mapeamento nome → função
TOOLS_MAP = {
    "get_timestamp": get_timestamp,
    "list_videos": list_videos,
}


def call_with_tools(query: str, model: str = None) -> dict:
    """
    Faz uma chamada ao LLM com function-calling habilitado.
    
    O LLM pode decidir chamar get_timestamp ou list_videos
    quando achar necessário, e depois gerar a resposta final.
    
    Args:
        query: pergunta do usuário
        model: modelo Groq
    
    Returns:
        Dict com: answer, tool_calls, model, tokens_used
    """
    if model is None:
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    client = get_groq_client()
    
    messages = [
        {
            "role": "system",
            "content": (
                "Você é um assistente de busca em podcasts/vídeos. "
                "Use a tool get_timestamp quando o usuário quiser encontrar "
                "um trecho específico. Use list_videos para mostrar o catálogo. "
                "Responda sempre em português do Brasil."
            ),
        },
        {"role": "user", "content": query},
    ]
    
    # Primeira chamada — LLM decide se usa tool
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS_DEFINITION,
        tool_choice="auto",
        temperature=0.3,
        max_tokens=1024,
    )
    
    message = response.choices[0].message
    tool_calls_made = []
    
    # Se o LLM decidiu usar tools
    if message.tool_calls:
        messages.append(message)
        
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # Executar a tool
            if function_name in TOOLS_MAP:
                result = TOOLS_MAP[function_name](**function_args)
            else:
                result = {"error": f"Tool '{function_name}' não encontrada."}
            
            tool_calls_made.append({
                "tool": function_name,
                "args": function_args,
                "result": result,
            })
            
            # Adicionar resultado da tool ao histórico
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })
        
        # Segunda chamada — LLM gera resposta final com dados da tool
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )
        
        answer = response.choices[0].message.content
    else:
        answer = message.content
    
    tokens_used = response.usage.total_tokens if response.usage else 0
    
    return {
        "answer": answer,
        "tool_calls": tool_calls_made,
        "model": model,
        "tokens_used": tokens_used,
    }


if __name__ == "__main__":
    # Teste rápido
    print("🔧 Testando tools...\n")
    
    # Teste list_videos
    print("📋 Vídeos disponíveis:")
    videos = list_videos()
    for v in videos.get("videos", []):
        print(f"   - {v['title']}")
    
    print()
    
    # Teste get_timestamp
    print("🔍 Buscando timestamp para 'machine learning':")
    result = get_timestamp("machine learning")
    if result.get("found"):
        print(f"   ⏱️  {result['timestamp']}")
        print(f"   📺 {result['video_title']}")
        print(f"   🔗 {result['url']}")
    else:
        print(f"   ❌ {result['message']}")
