from kogito.core.knowledge import KnowledgeGraph
from kogito.models.bart.comet import COMETBART

input_graph = KnowledgeGraph.from_jsonl("test_atomic2020_sample.json")

model = COMETBART.from_pretrained("mismayil/comet-bart-ai2")
scores = model.evaluate(
    input_graph, batch_size=256, num_return_sequences=1
)
print(scores)
