from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM


# 1. Define LLM 
llm = LLM(
    provider="google",
    model="gemini-2.5-flash"
)


# 2. Agent Definition
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


def run_chat():
    print("Elderly Nutrition Chatbot (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # 3. Define Task instead of tasks.yaml
        nutrition_task = Task(
            description=(
                f"You are given a question from a user: {user_input}.\n\n"
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

        print("\nBot:", result, "\n")


if __name__ == "__main__":
    run_chat()
