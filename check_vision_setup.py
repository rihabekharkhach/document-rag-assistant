import os

from app.agents.graph import research_graph
from app.rag import vision

print("env VISION_ENABLED =", repr(os.getenv("VISION_ENABLED")), "| module flag:", vision.VISION_ENABLED)
print("VLM_MODEL:", vision.VLM_MODEL, "| Google key set:", bool(os.getenv("GOOGLE_API_KEY")))

graph = research_graph.get_graph()
print("nodes:", list(graph.nodes))
for edge in graph.edges:
    print("edge:", edge.source, "->", edge.target, "(conditional)" if edge.conditional else "")