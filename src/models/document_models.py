from pydantic import BaseModel


class SearchResult(BaseModel):
    contexts: list[str]
    sources: list[str]
