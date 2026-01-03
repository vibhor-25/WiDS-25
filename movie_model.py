from pydantic import BaseModel, Field
from typing import List, Optional


# ---------- Movie Search Output ----------
class MovieSearchItem(BaseModel):
    title: str
    year: Optional[str] = None
    type: Optional[str] = None


class MovieSearchResult(BaseModel):
    query: str
    results: List[MovieSearchItem] = Field(default_factory=list)


# ---------- Movie Recommendation Output ----------
class MovieRecommendation(BaseModel):
    title: str
    reason: str


class MovieRecommendations(BaseModel):
    recommendations: List[MovieRecommendation] = Field(default_factory=list)


# ---------- Movie Detail Output ----------
class MovieDetail(BaseModel):
    title: str
    director: Optional[str] = None
    cast: Optional[str] = None
    plot: Optional[str] = None
    awards: Optional[str] = None
