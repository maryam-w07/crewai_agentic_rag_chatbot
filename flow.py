from crewai.flow.flow import Flow, start, router, listen
from crewai_tools import MCPServerAdapter
from pydantic import BaseModel


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
        context = query_knowledge_base.run(query=self.state.query)        # local KB
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
