import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
uv add streamlit-authenticator

# page layout

st.set_page_config(
    page_title="Nutrition Assistant",
    page_icon="🌿",
    layout="centered",
)

#configs

COLLECTION_NAME = "nutrition_knowledge"
QDRANT_PATH     = "./qdrant_storage"
TOP_K_RETRIEVE  = 20
TOP_K_RERANK    = 5

# load all the models once

@st.cache_resource
def load_models():
    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    qdrant   = QdrantClient(path=QDRANT_PATH)
    llm      = LLM(provider="google", model="gemini-2.5-flash")
    agent    = Agent(
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
    return embedder, reranker, qdrant, agent

embedder, reranker, qdrant, agent = load_models()

# session state

if "messages" not in st.session_state:
    st.session_state.messages = []

# Retrievel step

def retrieve(query):
    query_vector = embedder.encode(query, normalize_embeddings=True).tolist()

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K_RETRIEVE,
        with_payload=True,
    ).points

    if not results:
        return "No relevant information found in the knowledge base."

    candidates = [r.payload["text"] for r in results]
    payloads   = [r.payload          for r in results]

    scores  = reranker.predict([(query, chunk) for chunk in candidates])
    ranked  = sorted(zip(scores, candidates, payloads), reverse=True)
    top     = ranked[:TOP_K_RERANK]

    context_parts = []
    for i, (score, text, payload) in enumerate(top, 1):
        source = payload.get("source", "unknown")
        page   = payload.get("page", "?")
        context_parts.append(f"[Source {i}: {source}, page {page}]\n{text}")

    return "\n\n---\n\n".join(context_parts)

# chat history formatting

def format_history():
    if not st.session_state.messages:
        return "No previous conversation."
    lines = []
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)

# UI

st.title("Nutrition Assistant")
st.caption("Providing evidence-based nutritional guidance for older adults.")
st.divider()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask a nutrition question...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            context = retrieve(user_input)

            task = Task(
                description=(
                    f"=== Conversation History ===\n"
                    f"{format_history()}\n\n"
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
                    "A clear, concise, and helpful nutrition response tailored for elderly "
                    "individuals, with source citations where applicable."
                ),
                agent=agent,
            )

            crew     = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
            result   = crew.kickoff()
            response = str(result)

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
