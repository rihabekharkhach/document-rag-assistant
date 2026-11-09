import os
import sys
import types

os.environ["LLM_PROVIDER"] = "ollama"  # building the client needs neither a key nor a server
os.environ["LLM_MODEL"] = "test-model"
os.environ["VISION_ENABLED"] = "true"
os.environ.pop("JINA_API_KEY", None)

stub = types.ModuleType("app.rag.embeddings")
stub.EMBEDDING_DIM = 8
stub.embed_text = lambda text: [0.0] * 8
stub.embed_texts = lambda texts: [[0.0] * 8 for _ in texts]
sys.modules["app.rag.embeddings"] = stub