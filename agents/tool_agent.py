""" Agent that exevite tools via MCP """

from integrations.base_agent import BaseAgent 
from configs.settings import settings
from schemas.agents.tool_agent import AgentOutput

tool_agent = BaseAgent(
  system_prompt=settings.tool_agent_prompt_path.read_text(encoding="utf-8"),
  response_model=AgentOutput
)

def invoke_tool_agent(state):

  if isinstance(state, dict):
    tool_agent_response = tool_agent.invoke(state)

    if tool_agent_response:
      return tool_agent_response

  return f"\nProvided state to tool_agent is not dict."

  
  