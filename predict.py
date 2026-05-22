"""
Replicate Cog предиктор.

Деплой:
    cog push r8.im/<your-username>/song-finder
"""
from typing import Any
from cog import BasePredictor
from app.search import SongSearcher


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.searcher = SongSearcher()

    def run(self, query: str = "Sad love song about missing someone", top_k: int = 3) -> Any:
        return self.searcher.search(query, top_k=top_k)
