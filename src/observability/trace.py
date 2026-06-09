"""
trace.py — Logging estruturado para observabilidade.

Registra cada chamada ao pipeline com:
- Timestamp, query, model usado
- Tokens, latência, cache hit/miss
- Custo estimado

Logs salvos em: logs/pipeline.log
"""

import json
import logging
import time
from datetime import datetime
from functools import wraps
from pathlib import Path


# ============================================
# SETUP DO LOGGER
# ============================================

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Logger principal
logger = logging.getLogger("podcast_qa")
logger.setLevel(logging.INFO)

# Handler para arquivo (JSON lines)
file_handler = logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)

# Handler para console (formato legível)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)

# Formatters
class JsonFormatter(logging.Formatter):
    """Formata logs como JSON para fácil parsing."""
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)
        return json.dumps(log_data, ensure_ascii=False)

file_handler.setFormatter(JsonFormatter())
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logger.addHandler(file_handler)
logger.addHandler(console_handler)


# ============================================
# FUNÇÕES DE TRACING
# ============================================

def log_query(query: str, model: str, tokens: int, latency: float,
              cache_hit: bool = False, tool_used: str = None,
              cost: float = None):
    """
    Registra uma consulta ao pipeline.
    
    Args:
        query: pergunta do usuário
        model: modelo usado
        tokens: tokens consumidos
        latency: tempo de resposta (segundos)
        cache_hit: se foi respondido pelo cache
        tool_used: nome da tool usada (se houver)
        cost: custo estimado (USD)
    """
    extra = {
        "query": query[:200],
        "model": model,
        "tokens_used": tokens,
        "latency_ms": round(latency * 1000, 2),
        "cache_hit": cache_hit,
        "tool_used": tool_used,
        "cost_usd": cost,
    }
    
    record = logger.makeRecord(
        name="podcast_qa",
        level=logging.INFO,
        fn="",
        lno=0,
        msg=f"Query processed: {query[:50]}...",
        args=(),
        exc_info=None,
    )
    record.extra_data = extra
    logger.handle(record)


def timed(func):
    """Decorator que mede o tempo de execução."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        
        # Adicionar latência ao resultado se for dict
        if isinstance(result, dict):
            result["latency_seconds"] = round(elapsed, 3)
        
        return result
    return wrapper


def get_recent_logs(n: int = 20) -> list[dict]:
    """
    Retorna os N logs mais recentes.
    
    Returns:
        Lista de dicts com dados de cada log
    """
    log_file = LOG_DIR / "pipeline.log"
    
    if not log_file.exists():
        return []
    
    logs = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    return logs[-n:]


if __name__ == "__main__":
    print("🧪 Testando logging...\n")
    
    # Log de exemplo
    log_query(
        query="O que é machine learning?",
        model="llama-3.3-70b-versatile",
        tokens=500,
        latency=1.2,
        cache_hit=False,
        tool_used=None,
        cost=0.0005,
    )
    
    log_query(
        query="Me explica machine learning",
        model="cached",
        tokens=0,
        latency=0.05,
        cache_hit=True,
    )
    
    print("✅ Logs salvos em logs/pipeline.log")
    
    # Mostrar logs recentes
    recent = get_recent_logs(5)
    for log in recent:
        print(json.dumps(log, indent=2, ensure_ascii=False))
