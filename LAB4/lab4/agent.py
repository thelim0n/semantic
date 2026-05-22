import os
import sys

from dotenv import load_dotenv

from langchain.agents import create_agent

from langchain_core.messages import (
    HumanMessage,
    SystemMessage
)

from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

from prompts import BASE_SYSTEM_PROMPT
from schemas import AgentResponse
from skills import get_skill_prompt

load_dotenv()

class AgentService:

    def __init__(self):

        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.0
        )

        self.structured_llm = self.llm.with_structured_output(
            AgentResponse
        )

        self.mcp_client = MultiServerMCPClient(
            {
                "marketing_tools": {
                    "command": sys.executable,
                    "args": ["mcp_server.py"],
                    "transport": "stdio",
                }
            }
        )
        self._tools = None

    async def get_tools(self):
        if self._tools is None:
            self._tools = await self.mcp_client.get_tools()
        return self._tools

    async def build_agent(self):
        tools = await self.get_tools()

        return create_agent(
            model=self.llm,
            tools=tools,
        )

    async def generate_answer(
        self,
        user_input: str
    ):

        skill_prompt = get_skill_prompt(user_input)

        system_prompt = f"""
            {BASE_SYSTEM_PROMPT}

            ACTIVE SKILL:
            {skill_prompt}
        """

        agent = await self.build_agent()

        result = await agent.ainvoke(
            {
                "messages": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_input)
                ]
            }
        )

        final_text = result["messages"][-1].content

        parsed_response = await self.structured_llm.ainvoke(
            final_text
        )

        return parsed_response

agent_service = AgentService()