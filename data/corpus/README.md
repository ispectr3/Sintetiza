# Corpus — Transcrições de Vídeos

Esta pasta contém as transcrições dos vídeos do YouTube usados como corpus para o RAG.

## Como gerar

```bash
python -m src.pipeline.ingest
```

Os arquivos JSON são gerados automaticamente com formato:
```json
{
    "video_id": "abc123",
    "video_title": "Título do Vídeo",
    "video_url": "https://youtube.com/watch?v=abc123",
    "segments": [
        {"text": "texto do segmento", "start": 0.0, "duration": 5.0}
    ]
}
```

## Vídeos incluídos

1. O que é Machine Learning? (Código Fonte TV)
2. O que é Inteligência Artificial? (Código Fonte TV)
3. O que são Microsserviços? (Código Fonte TV)
4. O que é API? (Código Fonte TV)
5. Docker Tutorial (Código Fonte TV)
