"""
Module containing add customized pipeline implementation
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import sklearn.pipeline


class Pipeline(sklearn.pipeline.Pipeline):
    def get_stats(self) -> List[Tuple[str, Dict[str, Any]]]:
        stats = []
        for step in self.steps:
            name, transformer = step
            if hasattr(transformer, "get_stats"):
                stats.append((name, transformer.get_stats()))
        return stats

    def save_stats(self, filename: Union[str, Path]):
        stats = self.get_stats()
        with open(Path(filename), "w") as f:
            f.write(json.dumps(stats))
