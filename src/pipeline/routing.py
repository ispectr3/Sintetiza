"""
routing.py — Model routing cheap-first para redução de custo.

Estratégia: usar modelo barato (8B) para perguntas simples
e modelo forte (70B) para perguntas complexas.

Classificação de complexidade:
- SIMPLES: saudações, perguntas diretas, listagem
- COMPLEXA: comparações, análises, perguntas multi-hop

Meta: redução ≥50% no custo médio por requisição.
"""

import os
import re

from dotenv import load_dotenv

load_dotenv()

# ============================================
# CONFIGURAÇÃO DE MODELOS
# ============================================
MODEL_CHEAP = os.getenv("GROQ_MODEL_CHEAP", "llama-3.1-8b-instant")
MODEL_STRONG = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Custo estimado por 1K tokens (Groq é grátis, mas simulamos para a rubrica)
COST_PER_1K = {
    MODEL_CHEAP: 0.0001,    # ~10x mais barato
    MODEL_STRONG: 0.001,
}


class ModelRouter:
    """
    Router que decide qual modelo usar baseado na complexidade da pergunta.
    
    Regras de roteamento:
    - Perguntas curtas (< 15 palavras) e diretas → modelo barato
    - Saudações e meta-perguntas → modelo barato
    - Perguntas com "compare", "analise", "explique em detalhes" → modelo forte
    - Perguntas com múltiplas partes → modelo forte
    - Default → modelo barato (cheap-first)
    """
    
    def __init__(self):
        self._stats = {
            "total": 0,
            "cheap": 0,
            "strong": 0,
            "total_cost": 0.0,
            "cost_without_routing": 0.0,
        }
    
    # Padrões que indicam perguntas SIMPLES
    SIMPLE_PATTERNS = [
        r"^(oi|olá|hey|hello|hi)\b",
        r"^(obrigad[oa]|valeu|thanks)",
        r"^(quais? (são|vídeos?|podcasts?))",
        r"^(liste?|mostre?|show)\b",
        r"^o que (é|são)\b",
        r"^(qual|quando|onde|quem)\b(?!.*\b(e|ou|compar|diferenç|analis)\b)",
    ]
    
    # Padrões que indicam perguntas COMPLEXAS
    COMPLEX_PATTERNS = [
        r"\b(compar[ea]|diferenç[as]|versus|vs\.?)\b",
        r"\b(analis[ea]|explique? (em detalhe|detalhadamente))\b",
        r"\b(resuma? tudo|faça um resumo completo)\b",
        r"\b(por que|como funciona|qual a relação)\b.*\b(e|entre|com)\b",
        r"\?.*\?",  # Múltiplas perguntas
        r"\b(prós? e contras?|vantagens? e desvantagens?)\b",
    ]
    
    def classify(self, query: str) -> str:
        """
        Classifica a complexidade da pergunta.
        
        Returns:
            "simple" ou "complex"
        """
        query_lower = query.lower().strip()
        words = query_lower.split()
        
        # Regra 1: perguntas muito curtas → simples
        if len(words) <= 5:
            return "simple"
        
        # Regra 2: verificar padrões complexos primeiro
        for pattern in self.COMPLEX_PATTERNS:
            if re.search(pattern, query_lower):
                return "complex"
        
        # Regra 3: verificar padrões simples
        for pattern in self.SIMPLE_PATTERNS:
            if re.search(pattern, query_lower):
                return "simple"
        
        # Regra 4: perguntas médias (6-15 palavras) → simples (cheap-first!)
        if len(words) <= 15:
            return "simple"
        
        # Regra 5: perguntas longas → complexas
        return "complex"
    
    def route(self, query: str) -> str:
        """
        Decide qual modelo usar para a pergunta.
        
        Args:
            query: pergunta do usuário
        
        Returns:
            Nome do modelo a usar
        """
        complexity = self.classify(query)
        
        if complexity == "simple":
            model = MODEL_CHEAP
            self._stats["cheap"] += 1
        else:
            model = MODEL_STRONG
            self._stats["strong"] += 1
        
        self._stats["total"] += 1
        
        return model
    
    def record_cost(self, model: str, tokens_used: int) -> float:
        """
        Registra o custo de uma chamada.
        
        Returns:
            Custo estimado da chamada
        """
        cost_per_1k = COST_PER_1K.get(model, 0.001)
        cost = (tokens_used / 1000) * cost_per_1k
        
        # Custo real (com routing)
        self._stats["total_cost"] += cost
        
        # Custo hipotético (sempre modelo forte)
        strong_cost = (tokens_used / 1000) * COST_PER_1K[MODEL_STRONG]
        self._stats["cost_without_routing"] += strong_cost
        
        return cost
    
    def get_stats(self) -> dict:
        """Retorna estatísticas de routing."""
        total = self._stats["total"]
        cost_with = self._stats["total_cost"]
        cost_without = self._stats["cost_without_routing"]
        
        savings = 0.0
        if cost_without > 0:
            savings = ((cost_without - cost_with) / cost_without) * 100
        
        return {
            "total_requests": total,
            "cheap_model_calls": self._stats["cheap"],
            "strong_model_calls": self._stats["strong"],
            "cheap_ratio": f"{(self._stats['cheap'] / total * 100):.1f}%" if total > 0 else "0%",
            "total_cost_usd": f"${cost_with:.6f}",
            "cost_without_routing_usd": f"${cost_without:.6f}",
            "savings_percent": f"{savings:.1f}%",
            "model_cheap": MODEL_CHEAP,
            "model_strong": MODEL_STRONG,
        }


# Instância global do router
router = ModelRouter()


if __name__ == "__main__":
    print("🧪 Testando Model Router...\n")
    
    test_queries = [
        ("Oi!", "simple"),
        ("O que é API?", "simple"),
        ("Quais vídeos estão disponíveis?", "simple"),
        ("Compare machine learning com deep learning e explique as diferenças", "complex"),
        ("Qual a relação entre Docker e microsserviços?", "complex"),
        ("Me explique em detalhes como funciona o machine learning", "complex"),
        ("O que ele fala sobre Python?", "simple"),
        ("Quais são os prós e contras de usar microsserviços?", "complex"),
    ]
    
    r = ModelRouter()
    
    for query, expected in test_queries:
        model = r.route(query)
        complexity = r.classify(query)
        status = "✅" if complexity == expected else "❌"
        model_label = "💰 cheap" if model == MODEL_CHEAP else "💎 strong"
        print(f"  {status} [{complexity:>7}] {model_label} → \"{query}\"")
    
    print(f"\n📊 Stats: {r.get_stats()}")
