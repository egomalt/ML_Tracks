"""
Replicate Cog предиктор.

Деплой:
    cog push r8.im/<your-username>/song-finder
"""
from typing import Any
import cog
from app.search import SongSearcher


class Predictor(cog.BasePredictor):
    def setup(self) -> None:
        self.searcher = SongSearcher()

    def predict(
        self,
        query: str = cog.Input(
            description="Опишите песню, которую хотите найти (любой язык)",
            default="Грустная песня о любви и расставании",
        ),
        top_k: int = cog.Input(
            description="Количество результатов",
            default=3,
            ge=1,
            le=10,
        ),
    ) -> Any:
        results = self.searcher.search(query, top_k=top_k)
        return results
