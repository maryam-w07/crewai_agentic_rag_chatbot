import os
import streamlit as st
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
from crewai.tools import tool # Built-in decorator to convert Python functions to tools
from crewai_tools import EXASearchTool
from crewai_tools import MCPServerAdapter

load_dotenv()
OPENAI_API_KEY2 = os.getenv("OPENAI_API_KEY2")
exa_api_key= os.getenv('EXA_API_key')

# Page configuration
st.set_page_config(
    page_title="Nutrition Assistant",
    page_icon="🌿",
    layout="centered",
)

# Configuration Parameters
COLLECTION_NAME = "nutrition_knowledge"
TOP_K_RETRIEVE  = 20
TOP_K_RERANK    = 5

# Cache models to maximize performance across concurrent Streamlit loops
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    llm = LLM(model="gpt-4o-mini", api_key=OPENAI_API_KEY2)
    return embedder, reranker, llm

embedder, reranker, llm = load_models()

# Initialize Qdrant Client in local server mode
qdrant = QdrantClient(url="http://localhost:6333")

# Initialize Exa Search Tool
#exa_tool = EXASearchTool()

# --- NEW: EXPLICIT CUSTOM RAG TOOL DEFINITION ---
@tool("Query Local Nutrition Knowledge Base")
def query_knowledge_base(query: str) -> str:
    """
    Searches the internal medical and nutritional guidelines database for documents 
    relevant to the user's specific query. Use this tool whenever an explicit facts-based 
    nutrition, diet, or senior health question is asked.
    """
    # 1. Generate Query Vector
    query_vector = embedder.encode(query, normalize_embeddings=True).tolist()

    # 2. Retrieve Point candidates from Qdrant
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K_RETRIEVE,
        with_payload=True,
    ).points

    if not results:
        return "No local knowledge matching this query was found."

    candidates = [r.payload["text"] for r in results]
    payloads   = [r.payload for r in results]

    # 3. Apply CrossEncoder Reranking
    scores = reranker.predict([(query, chunk) for chunk in candidates])
    ranked = sorted(zip(scores, candidates, payloads), reverse=True)
    top    = ranked[:TOP_K_RERANK]

    context_parts = [text for _, text, _ in top]
    return "\n\n".join(context_parts)


# Construct the parameters using the retrieved key
exa_params = {
    "url": f"https://mcp.exa.ai/mcp?exaApiKey={exa_api_key}&tools=web_search_exa",
    "transport": "streamable-http",
}

# Define our specialized Agent with access to both Local Vector Storage & External Web Search via mcp using 'with' context
with MCPServerAdapter(exa_params) as exa_tools:
    elderly_diet_agent = Agent(
        role="Elderly Nutrition Assistant",
        goal="Answer nutrition and diet questions for elderly individuals concisely and safely.",
        backstory=(
            "You are an elderly nutrition assistant. You only answer questions related to "
            "food, diet, vitamins, hydration, and nutritional health for older adults.\n\n"
            "CRITICAL RULES:\n"
            "1. For any greeting or small talk (e.g., 'hi', 'hello', 'how are you'), do NOT call any tools. "
            "   Simply reply directly using a single friendly sentence.\n"
            "2. For nutrition questions, always run the 'Query Local Nutrition Knowledge Base' tool FIRST.\n"
            "3. If the local knowledge base tool returns insufficient or missing context for a nutrition topic, "
            "   only then utilize the 'web_search_exa' mcp tool to find answers on the live web.\n"
            "4. If a query is entirely unrelated to health, food, or senior care, do not call any tools. "
            "   Politely refuse to answer.\n"
            "5. Keep responses short, clear, and structured in bullet points.\n"
            "6. Never mention specific document names, file systems, page numbers, or tool outputs."
        ),
        tools=[query_knowledge_base,*exa_tools], # Both tools are now mapped natively here
        llm=llm,
        verbose=True,
    )

# Maintain persistent session state for Streamlit messaging loops
if "messages" not in st.session_state:
    st.session_state.messages = []

def format_history():
    if not st.session_state.messages:
        return "No previous conversation."
    lines = []
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)

# Render UI Layout
st.title("Nutrition Assistant")
st.caption("Providing evidence-based nutritional guidance for older adults using autonomous retrieval.")
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
        with st.spinner("Processing intent and generating response..."):
            
            # Formulate the localized operational Task 
            task = Task(
                description=(
                    f"Conversation history:\n{format_history()}\n\n"
                    f"Current User Input: {user_input}\n\n"
                    "Determine the intent. Run the appropriate tools only if necessary "
                    "to respond accurately. Strictly follow your backstory personas and formatting limitations."
                ),
                expected_output=(
                    "A direct, context-driven response matching the user's intent. "
                    "Bullet points for nutrition answers, single clear sentences for chit-chat/refusals."
                ),
                agent=elderly_diet_agent,
            )

            # Fire off execution
            crew     = Crew(agents=[elderly_diet_agent], tasks=[task], process=Process.sequential, verbose=False)
            result   = crew.kickoff()
            response = str(result)

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
