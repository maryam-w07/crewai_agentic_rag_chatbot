import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
from crewai_tools import EXASearchTool
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY2 = os.getenv("OPENAI_API_KEY2")

# Page config 

st.set_page_config(
    page_title="Nutrition Assistant",
    page_icon="🌿",
    layout="centered",
)

# Exa Search Tool

#Config 

COLLECTION_NAME = "nutrition_knowledge"
QDRANT_PATH     = "./qdrant_storage"
TOP_K_RETRIEVE  = 20
TOP_K_RERANK    = 5

# load models once
print(OPENAI_API_KEY2[:10])

@st.cache_resource
def load_models():
    embedder   = SentenceTransformer("BAAI/bge-base-en-v1.5")
    reranker   = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    
    llm = LLM(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY2
        
       )
    return embedder, reranker, llm

embedder, reranker, llm = load_models()

# qdrant (server mode for concurrent sessions) 

qdrant = QdrantClient(url="http://localhost:6333")

# agent

elderly_diet_agent = Agent(
    role="Geriatric Nutrition Specialist",
    goal="Answer nutrition and diet questions for elderly individuals concisely and safely.",
    backstory=(
        "You are a geriatric nutrition specialist. You only answer questions related to "
        "food, diet, vitamins, hydration, and nutritional health for older adults. "
        "For topics not covered by your knowledge base, you use the web search tool "
        "to find relevant health and nutrition information. "
        "You never discuss unrelated topics. "
        "You always keep responses short, clear, and in bullet points. "
        "You never mention document names, sources, page numbers, or where information came from."
    ),
    tools=[exa_tool],
    llm=llm,
    verbose=True,
)

# session state

if "messages" not in st.session_state:
    st.session_state.messages = []

# Retrieval 

def retrieve(query):
    query_vector = embedder.encode(query, normalize_embeddings=True).tolist()

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K_RETRIEVE,
        with_payload=True,
    ).points

    if not results:
        return ""

    candidates = [r.payload["text"] for r in results]
    payloads   = [r.payload          for r in results]

    scores  = reranker.predict([(query, chunk) for chunk in candidates])
    ranked  = sorted(zip(scores, candidates, payloads), reverse=True)
    top     = ranked[:TOP_K_RERANK]

    context_parts = []
    for i, (score, text, payload) in enumerate(top, 1):
        context_parts.append(text)

    return "\n\n".join(context_parts)

#Chat history formatter 

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

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a nutrition question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            context = retrieve(user_input)

            task = Task(
                description=(
                    f"Conversation history:\n{format_history()}\n\n"
                    f"Knowledge base context (use if relevant, do not cite it):\n{context}\n\n"
                    f"User question: {user_input}\n\n"
                    "Instructions:\n"
                    "1. If this is a greeting or small talk, reply in one short friendly sentence only.\n"
                    "2. If this is a nutrition or diet question for elderly:\n"
                    "   - Check the knowledge base context first.\n"
                    "   - If context is insufficient, use the web search tool to find relevant information.\n"
                    "   - Respond in short bullet points only.\n"
                    "   - Never mention documents, sources, page numbers, or search results.\n"
                    "3. If this is unrelated to elderly nutrition or health, respond with:\n"
                    "   'My knowledge is limited to nutrition and diet topics for older adults.'\n"
                ),
                expected_output=(
                    "A short, concise response in bullet points for nutrition questions, "
                    "or a single sentence for greetings and off-topic queries. "
                    "No document references, no source mentions."
                ),
                agent=elderly_diet_agent,
            )

            crew     = Crew(agents=[elderly_diet_agent], tasks=[task], process=Process.sequential, verbose=False)
            result   = crew.kickoff()
            response = str(result)

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
