"""
test_smoke.py — Smoke tests para validar que o pipeline funciona.

Roda sem API key (testa apenas componentes locais).
Para rodar: python -m pytest tests/test_smoke.py -v
"""

import json
import os
import sys
from pathlib import Path

# Adicionar raiz ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_corpus_exists():
    """Verifica se existem transcrições no corpus."""
    corpus_dir = Path("data/corpus/transcripts")
    assert corpus_dir.exists(), f"Pasta {corpus_dir} não existe! Rode: python -m src.pipeline.ingest"
    
    json_files = list(corpus_dir.glob("*.json"))
    assert len(json_files) >= 1, f"Nenhuma transcrição encontrada em {corpus_dir}/"
    
    # Verificar que cada arquivo é JSON válido
    for f in json_files:
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        assert "video_id" in data, f"Arquivo {f.name} não tem 'video_id'"
        assert "segments" in data, f"Arquivo {f.name} não tem 'segments'"
        assert len(data["segments"]) > 0, f"Arquivo {f.name} não tem segmentos"
    
    print(f"✅ {len(json_files)} transcrições válidas encontradas")


def test_chunking():
    """Verifica que o chunking funciona corretamente."""
    from src.pipeline.rag import chunk_transcript
    
    # Dados de teste
    fake_transcript = {
        "video_id": "test123",
        "video_title": "Vídeo de Teste",
        "video_url": "https://youtube.com/watch?v=test123",
        "segments": [
            {"text": f"Segmento {i} com texto de teste para chunking.", "start": i * 10.0, "duration": 9.5}
            for i in range(50)
        ],
    }
    
    chunks = chunk_transcript(fake_transcript)
    
    assert len(chunks) > 0, "Nenhum chunk gerado!"
    assert all("text" in c for c in chunks), "Chunks sem campo 'text'"
    assert all("video_id" in c for c in chunks), "Chunks sem campo 'video_id'"
    assert all("start_time" in c for c in chunks), "Chunks sem campo 'start_time'"
    assert all("end_time" in c for c in chunks), "Chunks sem campo 'end_time'"
    
    # Verificar que timestamps estão presentes
    assert chunks[0]["start_time"] >= 0, "start_time negativo"
    
    print(f"✅ Chunking OK: {len(chunks)} chunks gerados a partir de 50 segmentos")


def test_chroma_connection():
    """Verifica que o ChromaDB está acessível."""
    import chromadb
    
    client = chromadb.PersistentClient(path="chroma_db")
    
    # Criar collection de teste
    test_col = client.get_or_create_collection("test_smoke")
    test_col.add(
        ids=["test_1"],
        documents=["Texto de teste para smoke test"],
        metadatas=[{"source": "test"}],
    )
    
    # Query
    results = test_col.query(query_texts=["teste"], n_results=1)
    assert len(results["ids"][0]) == 1, "Query no ChromaDB falhou"
    
    # Cleanup
    client.delete_collection("test_smoke")
    
    print("✅ ChromaDB conectado e funcionando")


def test_model_router():
    """Verifica que o model router classifica corretamente."""
    from src.pipeline.routing import ModelRouter, MODEL_CHEAP, MODEL_STRONG
    
    router = ModelRouter()
    
    # Perguntas simples → modelo barato
    assert router.route("Oi!") == MODEL_CHEAP
    assert router.route("O que é API?") == MODEL_CHEAP
    assert router.route("Quais vídeos?") == MODEL_CHEAP
    
    # Perguntas complexas → modelo forte
    assert router.route("Compare machine learning com deep learning e explique as diferenças") == MODEL_STRONG
    
    print("✅ Model Router OK")


def test_semantic_cache():
    """Verifica que o cache semântico funciona."""
    from src.pipeline.cache import SemanticCache
    
    cache = SemanticCache()
    
    # Lookup em cache vazio → miss
    result = cache.lookup("pergunta teste")
    assert result is None, "Cache deveria estar vazio"
    
    # Store
    cache.store("pergunta teste", {
        "answer": "resposta teste",
        "model": "test",
        "tokens_used": 100,
        "sources": [],
    })
    
    # Lookup exato → hit
    result = cache.lookup("pergunta teste")
    assert result is not None, "Cache deveria ter a resposta"
    assert result["answer"] == "resposta teste"
    
    # Stats
    stats = cache.get_stats()
    assert stats["cache_hits"] >= 1, "Deveria ter pelo menos 1 hit"
    
    # Cleanup
    cache.clear()
    
    print("✅ Cache Semântico OK")


def test_env_file():
    """Verifica que o .env.example existe."""
    env_example = Path(".env.example")
    assert env_example.exists(), ".env.example não encontrado!"
    
    content = env_example.read_text()
    assert "GROQ_API_KEY" in content, ".env.example não tem GROQ_API_KEY"
    
    print("✅ .env.example OK")


def test_tools_definition():
    """Verifica que as tools estão definidas corretamente."""
    from src.pipeline.tools import TOOLS_DEFINITION, TOOLS_MAP
    
    assert len(TOOLS_DEFINITION) >= 1, "Nenhuma tool definida"
    assert "get_timestamp" in TOOLS_MAP, "Tool get_timestamp não encontrada"
    
    # Verificar formato
    for tool_def in TOOLS_DEFINITION:
        assert tool_def["type"] == "function"
        assert "name" in tool_def["function"]
        assert "description" in tool_def["function"]
        assert "parameters" in tool_def["function"]
    
    print(f"✅ {len(TOOLS_DEFINITION)} tools definidas corretamente")


if __name__ == "__main__":
    print("=" * 50)
    print("🧪 SMOKE TESTS — Podcast Q&A Assistant")
    print("=" * 50)
    print()
    
    tests = [
        ("Env file", test_env_file),
        ("Chunking", test_chunking),
        ("ChromaDB", test_chroma_connection),
        ("Model Router", test_model_router),
        ("Semantic Cache", test_semantic_cache),
        ("Tools Definition", test_tools_definition),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"❌ {name}: {e}")
            failed += 1
    
    print()
    print("=" * 50)
    print(f"📊 Resultado: {passed} ✅ passed | {failed} ❌ failed")
    print("=" * 50)
    
    # Teste de corpus (opcional — só funciona depois do ingest)
    try:
        test_corpus_exists()
        passed += 1
    except AssertionError as e:
        print(f"⚠️  Corpus: {e} (rode python -m src.pipeline.ingest primeiro)")
