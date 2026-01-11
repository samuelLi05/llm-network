import os
import json
import random
from typing import List

"""Generates random prompts based on predefined templates and word lists.

All agents share a centralized discussion topic but each uses unique
randomly selected templates and wording.
"""

class PromptGenerator:
    def __init__(self, topic: str = None):
        """Initialize with a shared topic. If topic is None, one is randomly chosen."""
        prompt_path = os.path.join(os.path.dirname(__file__), 'random_prompt.json')
        with open(prompt_path, 'r') as f:
            self.prompts = json.load(f)
        # Shared topic across all agents
        self.topic = topic if topic else random.choice(self.prompts['topics'])

    def generate_single_prompt(self) -> str:
        """Generate a single prompt with random template and word substitutions."""
        template = random.choice(self.prompts['templates'])
        prompt = template
        for cat, words in self.prompts.items():
            if cat in ['topics', 'templates']:
                continue
            if isinstance(words, list) and f'{{{cat}}}' in prompt:
                prompt = prompt.replace(f'{{{cat}}}', random.choice(words))
        prompt = prompt.replace('{topic}', self.topic)
        return prompt
    
    def generate_multiple_prompts(self, n: int) -> List[str]:
        """Generate n unique prompts sharing the same topic."""
        return [self.generate_single_prompt() for _ in range(n)]
    
    def get_topic(self) -> str:
        """Return the shared discussion topic."""
        return self.topic

    