"""
cache.py — Cache semântico para redução de custo.

Quando o usuário faz uma pergunta parecida com uma anterior,
retorna a resposta do cache sem chamar a API do Groq.

Isso reduz:
- Custo (menos chamadas de API)
- Latência (resposta instantânea do cache)

Métricas reportadas:
- Cache hit-rate (% de perguntas respondidas pelo cache)
- Economia estimada de tokens
"""

import json
import os
import time
from pathlib import Path

import chromadb


# ============================================
# CONFIGURAÇÃO
# ============================================
CACHE_DB_PATH = "chroma_db"
CACHE_COLLECTION = "semantic_cache"
SIMILARITY_THRESHOLD = 0.92  # Quão similar precisa ser para usar cache (0-1)
CACHE_TTL = 3600  # Time-to-live em segundos (1 hora)


class SemanticCache:
    """
    Cache semântico usando ChromaDB para busca por similaridade.
    
    Funciona assim:
    1. Quando recebe uma pergunta, gera embedding e busca no cache
    2. Se encontrar pergunta similar (>92% similaridade) → retorna cache
    3. Se não encontrar → retorna None (pipeline RAG normal)
    4. Depois que o RAG responde, salva no cache para próximas vezes
    """
    
    def __init__(self):
        self._collection = None
        self._stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "tokens_saved": 0,
        }
    
    @property
    def collection(self):
        """Lazy loading da collection do cache."""
        if self._collection is None:
            client = chromadb.PersistentClient(path=CACHE_DB_PATH)
            self._collection = client.get_or_create_collection(
                name=CACHE_COLLECTION,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection
    
    def lookup(self, query: str) -> dict | None:
        """
        Busca uma resposta no cache para a pergunta.
        
        Args:
            query: pergunta do usuário
        
        Returns:
            Dict com resposta cacheada ou None se não encontrar
        """
        self._stats["total_queries"] += 1
        
        if self.collection.count() == 0:
            self._stats["cache_misses"] += 1
            return None
        
        # Buscar pergunta similar
        results = self.collection.query(
            query_texts=[query],
            n_results=1,
        )
        
        if not results["ids"][0]:
            self._stats["cache_misses"] += 1
            return None
        
        # Verificar similaridade (distance = 1 - similarity para cosine)
        distance = results["distances"][0][0]
        similarity = 1 - distance
        
        if similarity < SIMILARITY_THRESHOLD:
            self._stats["cache_misses"] += 1
            return None
        
        # Verificar TTL
        metadata = results["metadatas"][0][0]
        cached_at = metadata.get("cached_at", 0)
        
        if time.time() - cached_at > CACHE_TTL:
            # Cache expirado — remover
            self.collection.delete(ids=[results["ids"][0][0]])
            self._stats["cache_misses"] += 1
            return None
        
        # Cache hit!
        self._stats["cache_hits"] += 1
        tokens_saved = metadata.get("tokens_used", 0)
        self._stats["tokens_saved"] += tokens_saved
        
        # Reconstruir a resposta do cache
        cached_data = json.loads(metadata.get("response_json", "{}"))
        cached_data["cache_hit"] = True
        cached_data["similarity"] = round(similarity, 3)
        cached_data["original_query"] = metadata.get("original_query", query)
        
        return cached_data
    
    def store(self, query: str, response: dict) -> None:
        """
        Armazena uma resposta no cache.
        
        Args:
            query: pergunta original
            response: resposta do pipeline RAG
        """
        cache_id = f"cache_{int(time.time() * 1000)}"
        
        # Serializar resposta (sem campos que não são string/int/float)
        response_clean = {
            "answer": response.get("answer", ""),
            "model": response.get("model", ""),
            "tokens_used": response.get("tokens_used", 0),
            "sources": response.get("sources", []),
        }
        
        self.collection.add(
            ids=[cache_id],
            documents=[query],
            metadatas=[{
                "original_query": query,
                "cached_at": time.time(),
                "tokens_used": response.get("tokens_used", 0),
                "model": response.get("model", "unknown"),
                "response_json": json.dumps(response_clean, ensure_ascii=False),
            }],
        )
    
    def get_stats(self) -> dict:
        """Retorna estatísticas do cache."""
        total = self._stats["total_queries"]
        hits = self._stats["cache_hits"]
        
        return {
            "total_queries": total,
            "cache_hits": hits,
            "cache_misses": self._stats["cache_misses"],
            "hit_rate": f"{(hits / total * 100):.1f}%" if total > 0 else "0.0%",
            "hit_rate_float": hits / total if total > 0 else 0.0,
            "tokens_saved": self._stats["tokens_saved"],
            "cache_size": self.collection.count(),
            "ttl_seconds": CACHE_TTL,
            "similarity_threshold": SIMILARITY_THRESHOLD,
        }
    
    def clear(self) -> None:
        """Limpa todo o cache."""
        client = chromadb.PersistentClient(path=CACHE_DB_PATH)
        try:
            client.delete_collection(CACHE_COLLECTION)
        except Exception:
            pass
        self._collection = None
        self._stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "tokens_saved": 0,
        }


# Instância global do cache
cache = SemanticCache()


if __name__ == "__main__":
    print("🧪 Testando Cache Semântico...\n")
    
    # Simular uso
    test_cache = SemanticCache()
    
    # Primeira consulta — miss
    result = test_cache.lookup("O que é machine learning?")
    print(f"Lookup 1: {'HIT' if result else 'MISS'}")
    
    # Armazenar resposta fake
    test_cache.store("O que é machine learning?", {
        "answer": "Machine learning é...",
        "model": "llama-3.3-70b-versatile",
        "tokens_used": 500,
        "sources": [],
    })
    print("Armazenado no cache.")
    
    # Segunda consulta (similar) — hit
    result = test_cache.lookup("Me explica o que é machine learning")
    print(f"Lookup 2: {'HIT' if result else 'MISS'}")
    
    # Estatísticas
    stats = test_cache.get_stats()
    print(f"\n📊 Stats: {json.dumps(stats, indent=2)}")
