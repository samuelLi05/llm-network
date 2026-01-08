from typing import List
from agents.network_agent import NetworkAgent

"""Managees ordering and coordination of responses among multiple NetworkAgents."""
class OrderManager:
    def __init__(self, agents: List[NetworkAgent]):
        self.agents = agents
    #TODO: Implement ordering and coordination logic
    pass
