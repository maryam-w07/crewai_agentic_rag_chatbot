"""
chatbot.py  —  Advanced RAG Nutrition Chatbot

Pipeline: User query → BGE embed → Qdrant search (top 20)  → Cross-encoder rerank(top 5) → Gemini via CrewAI

Run:
    python chatbot.py
"""

from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM

# Configs
COLLECTION_NAME  = "nutrition_knowledge"
QDRANT_PATH      = "./qdrant_storage"
TOP_K_RETRIEVE   = 20    #how many chunks to fetch from Qdrant
TOP_K_RERANK     = 5     # how many chunks to pass to the LLM after reranking

#Load models 
print("Loading embedding model...")
embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")

print("Loading reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

print("Connecting to Qdrant...")
qdrant = QdrantClient(path=QDRANT_PATH)

#LLM 
llm = LLM(provider="google", model="gemini-2.5-flash")

# Agent
agent = Agent(
    role="Elderly Nutrition Assistant",
    goal="Provide clear, safe, and practical nutritional advice for elderly individuals.",
    backstory=(
        "You are a compassionate nutrition expert specialising in elderly care. "
        "You explain dietary concepts simply and clearly. "
        "You always base your answers on the provided context when available. "
        "You never give medical diagnoses — only general nutrition guidance."
    ),
    llm=llm,
    verbose=False,
)

#Retrieval 
def retrieve(query):
    # Step 1: Embed the query
    query_vector = embedder.encode(query, normalize_embeddings=True).tolist()

    #Step 2: Search Qdrant for top 20 similar chunks
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K_RETRIEVE,
        with_payload=True,
    ).points

    if not results:
        return "No relevant information found in the knowledge base."

    # Step 3: Rerank: cross-encoder scores each (query, chunk) pair
    candidates = [r.payload["text"] for r in results]
    payloads   = [r.payload          for r in results]

    scores  = reranker.predict([(query, chunk) for chunk in candidates])
    ranked  = sorted(zip(scores, candidates, payloads), reverse=True)
    top     = ranked[:TOP_K_RERANK]

    # Step 4: Format context with source info
    context_parts = []
    for i, (score, text, payload) in enumerate(top, 1):
        source = payload.get("source", "unknown")
        page   = payload.get("page", "?")
        context_parts.append(f"[Source {i}: {source}, page {page}]\n{text}")

    return "\n\n---\n\n".join(context_parts)

# Chat history
def format_history(history):
    if not history:
        return "No previous conversation."
    lines = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Bot"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)

#Main chat loop

def run_chat():
    print("\n Elderly Nutrition Chatbot  —  Advanced RAG")
    print("   Type 'exit' to quit.\n")

    history = []

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye.")
            break

        # Retrieve relevant context
        print("Retrieving context...")
        context = retrieve(user_input)

        # Build task
        task = Task(
            description=(
                f"=== Conversation History ===\n"
                f"{format_history(history)}\n\n"
                f"=== Retrieved Context ===\n"
                f"{context}\n\n"
                f"=== User Question ===\n"
                f"{user_input}\n\n"
                "Answer using the retrieved context above as your primary source. "
                "Cite the source label (e.g. Source 1) when referencing it. "
                "If the context does not cover the question, use your general knowledge "
                "and mention that. Keep the response clear and simple for elderly readers."
            ),
            expected_output=(
                "A clear, concise, and helpful nutrition response tailored for elderly individuals, "
                "with source citations where applicable."
            ),
            agent=agent,
        )

        # Run crew
        crew   = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
        result = crew.kickoff()
        response = str(result)

        print(f"\nBot: {response}\n")

        history.append({"role": "user", "content": user_input})
        history.append({"role": "bot",  "content": response})


if __name__ == "__main__":
    run_chat()
