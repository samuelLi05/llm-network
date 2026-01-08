import os
import json
import random
from typing import List

"""Generates random prompts based on predefined templates and word lists."""

class PromptGenerator:
    def __init__(self):
        prompt_path = os.path.join(os.path.dirname(__file__), 'random_prompts.json')
        self.prompts = json.load(open(prompt_path, 'r'))
        self.topic = random.choice(self.prompts['topics'])

    def generate_single_prompt(self) -> str:
        template = random.choice(self.prompts['templates'])
        for cat, word in self.prompts.items():
            if cat in ['topics', 'templates']:
                continue
            prompt = template.replace(f'{{{cat}}}', random.choice(word))
        return prompt.replace('{topic}', self.topic)
    
    def generate_multiple_prompts(self, n: int) -> List[str]:
        return [self.generate_single_prompt() for _ in range(n)]

    