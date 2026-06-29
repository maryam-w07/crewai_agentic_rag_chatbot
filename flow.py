import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from crewai.llm import LLM
from crewai.tools import tool
from crewai.flow.flow import Flow, start, router, listen
from crewai_tools import MCPServerAdapter
from pydantic import BaseModel

# 1. Load your environment keys
load_dotenv()
OPENAI_API_KEY2 = os.getenv("OPENAI_API_KEY2")
exa_api_key = os.getenv('EXA_API_key')

# 2. Configuration Parameters
COLLECTION_NAME = "nutrition_knowledge"
TOP_K_RETRIEVE  = 20
TOP_K_RERANK    = 5

# 3. Instantiate the LLM, Embedder, and Reranker
llm = LLM(model="gpt-4o-mini", api_key=OPENAI_API_KEY2)
embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

# 4. Connect to Qdrant
qdrant = QdrantClient(url="http://localhost:6333")

def query_knowledge_base(query: str) -> str:
    query_vector = embedder.encode(query, normalize_embeddings=True).tolist()
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

    scores = reranker.predict([(query, chunk) for chunk in candidates])
    ranked = sorted(zip(scores, candidates, payloads), reverse=True)
    top    = ranked[:TOP_K_RERANK]

    context_parts = [text for _, text, _ in top]
    return "\n\n".join(context_parts)

# 6. Configure Exa parameters
exa_params = {
    "url": f"https://mcp.exa.ai/mcp?exaApiKey={exa_api_key}&tools=web_search_exa",
    "transport": "streamable-http",
}

#flow logic
class ChatState(BaseModel):
    query: str = ""
    history: str = ""
    answer: str = ""

class NutritionFlow(Flow[ChatState]):

    @start()
    def receive_query(self):
        return self.state.query

    @router(receive_query)
    def classify_intent(self):
        decision = llm.call(
            "Classify the message as exactly one word — greeting, nutrition, or out_of_scope.\n\n"
            f"History:\n{self.state.history}\n\nMessage: {self.state.query}"
        ).strip().lower()
        if "greeting" in decision:  return "greeting"
        if "nutrition" in decision: return "nutrition"
        return "out_of_scope"

    @listen("greeting")
    def handle_greeting(self):
        self.state.answer = "Hello! Ask me anything about nutrition for older adults."
        return self.state.answer

    @listen("nutrition")
    def handle_nutrition(self):
        context = query_knowledge_base(self.state.query)       # local KB
        if context.startswith("No local knowledge"):                      #  Exa web (MCP)
            with MCPServerAdapter(exa_params) as exa_tools:
                context = exa_tools[0].run(query=self.state.query)
        self.state.answer = llm.call(
            "Answer using ONLY this context, in short bullets.\n\n"
            f"Context:\n{context}\n\nQuestion: {self.state.query}"
        )
        return self.state.answer

    @listen("out_of_scope")
    def handle_refusal(self):
        self.state.answer = "Sorry — I can only help with nutrition and senior-health questions."
        return self.state.answer
#testing
if __name__ == "__main__":
    # 1. Instantiate your flow
    flow = NutritionFlow()
    
    # 2. Set the initial query you want to test
    flow.state.query = "What are the best vitamins for a 75-year-old woman?"
    flow.state.history = "User: Hello\nAssistant: Hello! Ask me anything about nutrition for older adults."
    
    print(f"🚀 Kicking off Flow with query: '{flow.state.query}'...\n")
    
    # 3. Execute the flow
    final_answer = flow.kickoff()
    
    # 4. Print the output
    print("\n🏁 --- FINAL ANSWER --- 🏁")
    print(final_answer)
