from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent


@CrewBase
class CrewaiNutritionAssist():
    """CrewaiNutritionAssist crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def nutrition_expert(self) -> Agent:
        return Agent(
            config=self.agents_config['nutrition_expert'],
            llm=LLM(
                provider="google",
                model="gemini-1.5-flash"
            ),
            verbose=True
        )

    @task
    def nutrition_advice_task(self) -> Task:
        return Task(
            config=self.tasks_config['nutrition_advice_task'],
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )