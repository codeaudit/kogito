from kogito.core.knowledge import KnowledgeGraph
from kogito.models.gpt2.comet import COMETGPT2

input_graph = KnowledgeGraph.from_jsonl("test_atomic2020.json")

model = COMETGPT2.from_pretrained("mismayil/comet-bart-ai2")
scores = model.evaluate(input_graph)

print(scores)
