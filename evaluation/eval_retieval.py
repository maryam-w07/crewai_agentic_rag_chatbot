import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

COLLECTION_NAME = "nutrition_knowledge"
QDRANT_PATH = "./qdrant_storage"

TOP_K = 10

print("Loading embedding model...")
embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")

print("Connecting to Qdrant...")
client = QdrantClient(path=QDRANT_PATH)

print("Loading evaluation dataset...")
with open("eval_dataset.json", "r") as f:
    dataset = json.load(f)


def retrieve(query, k=TOP_K):
    vector = embedder.encode(query, normalize_embeddings=True).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=k,
        with_payload=True
    ).points

    return [r.id for r in results]


def precision_at_k(k):
    total = 0

    for item in dataset:
        gold = set(item["gold_chunk_ids"])
        retrieved = retrieve(item["question"], k)

        hits = len(gold.intersection(retrieved))
        total += hits / k

    return total / len(dataset)


def recall_at_k(k):
    total = 0

    for item in dataset:
        gold = set(item["gold_chunk_ids"])
        retrieved = retrieve(item["question"], k)

        hits = len(gold.intersection(retrieved))
        total += hits / len(gold)

    return total / len(dataset)


def mrr():
    total = 0

    for item in dataset:
        gold = set(item["gold_chunk_ids"])
        retrieved = retrieve(item["question"], TOP_K)

        reciprocal_rank = 0

        for rank, chunk_id in enumerate(retrieved, start=1):
            if chunk_id in gold:
                reciprocal_rank = 1 / rank
                break

        total += reciprocal_rank

    return total / len(dataset)


print("\nRunning retrieval evaluation...\n")

p5 = precision_at_k(5)
r5 = recall_at_k(5)
r10 = recall_at_k(10)
m = mrr()

print("=" * 40)
print(" Retrieval Evaluation Results")
print("=" * 40)

print(f"Precision@5 : {p5:.3f}")
print(f"Recall@5    : {r5:.3f}")
print(f"Recall@10   : {r10:.3f}")
print(f"MRR         : {m:.3f}")

print("=" * 40)
