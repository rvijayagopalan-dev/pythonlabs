import yaml
from pathlib import Path
from typing import Optional, Dict, Any

class PromptRegistry:
    def __init__(self, path: str = str(Path(__file__).with_name('prompts') / 'registry.yaml')):
        self.path = Path(path)
        with open(self.path, 'r', encoding='utf-8') as f:
            self._data = yaml.safe_load(f)

    def get(self, name: str, version: Optional[str] = None) -> Dict[str, Any]:
        node = self._data.get(name)
        if not node:
            raise KeyError(f"Prompt not found: {name}")
        if version is None:
            # pick the last key as latest by convention
            version = list(node.keys())[-1]
        return node[version]
