from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM


# 1. Define LLM 
llm = LLM(
    provider="google",
    model="gemini-2.5-flash"
)


# 2.Agent Definition
nutrition_agent = Agent(
    role="Elderly Nutrition Assistant",
    goal="Provide clear, safe, and practical nutritional advice tailored for elderly individuals.",
    backstory=(
        "You are a compassionate and knowledgeable nutrition expert specializing in elderly care. "
        "You simplify complex dietary concepts into easy-to-understand guidance. "
        "You focus on promoting healthy aging through balanced diets, hydration, and safe food choices. "
        "You avoid giving medical diagnoses and instead provide general wellness advice."
    ),
    llm=llm,
    verbose=True,
    #memory=True  #short-term mem(remembers within a run)/doesnt save or remember from past sessions.Lightweight internal agent memory for contextual reasoning within an execution.
)

#3. in-memory chat history/formatting
def format_history(conversation_history):
    """convert user-agent conversation history list into readable text prompt"""
    if not conversation_history:
        return "no previous conversation"
    
    formatted= ""
    for msg in conversation_history:
        role = "User" if msg["role"] == "user" else "bot"
        formatted += f"{role}: {msg['content']}\n"
    return formatted


def run_chat():
    print("Elderly Nutrition Chatbot (type 'exit' to quit)\n")
    conversation_history = [] #chat history RAM only

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # 3. Define Task instead of tasks.yaml
        nutrition_task = Task(
            description=(
                f"Previous conversation:\n{format_history(conversation_history)}\n\n"  # hist
                f"User's latest message: {user_input}\n\n"
                "Provide clear, simple, and safe nutritional advice specifically for elderly individuals.\n\n"
                "Focus on:\n"
                "- Easy-to-understand explanations\n"
                "- Practical dietary suggestions\n"
                "- General wellness tips (not medical diagnosis)\n"
            ),
            expected_output=(
                "A helpful and concise response that gives nutritional advice "
                "tailored to elderly individuals based on the user's question."
            ),
            agent=nutrition_agent
        )

        # 4. Define crew
        crew = Crew(
            agents=[nutrition_agent],
            tasks=[nutrition_task],
            process=Process.sequential,
            verbose=True
        )

        result = crew.kickoff()

        print("\nbot:", result, "\n") #result obj printed as it is
        bot_response = str(result)  #convert to string
        
        #save both messages to history AFTER each exchange
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "bot", "content": bot_response})


if __name__ == "__main__":
    run_chat()
