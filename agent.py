from google.adk.agents import Agent, LlmAgent
from google.adk.tools import google_search, AgentTool, ToolContext

from .movie_model import (
    MovieSearchResult,
    MovieRecommendations,
    MovieDetail,
)

import os
import requests
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
OMDB_API_KEY = os.getenv("MOVIE_API_KEY")
URL = f"https://www.omdbapi.com/?apikey={OMDB_API_KEY}&s="


# ==================================================
# TOOL: External Movie Search (NO STATE SIDE EFFECTS)
# ==================================================

def fetch_movies(query: str, tool_context: ToolContext) -> dict:
    response = requests.get(URL + query)
    data = response.json()

    return {
        "query": query,
        "results": [
            {
                "title": m.get("Title"),
                "year": m.get("Year"),
                "type": m.get("Type"),
            }
            for m in data.get("Search", [])
        ] if data.get("Response") == "True" else []
    }


# ==================================================
# TOOL: Centralized Persistent Memory Updater
# ==================================================

def update_memory(
    event_type: str,
    payload: dict,
    tool_context: ToolContext
) -> dict:
    """
    The ONLY place where persistent session memory is mutated.
    """

    # -------- SEARCH HISTORY --------
    if event_type == "search":
        history = tool_context.state.get("movie.search.history", [])
        history.append({
            "query": payload["query"],
            "timestamp": datetime.utcnow().isoformat()
        })
        tool_context.state["movie.search.history"] = history

    # -------- RECOMMENDATION HISTORY --------
    elif event_type == "recommendation":
        history = tool_context.state.get("movie.recommendations.history", [])
        history.append({
            "titles": payload["titles"],
            "timestamp": datetime.utcnow().isoformat()
        })
        tool_context.state["movie.recommendations.history"] = history

    # -------- USER PREFERENCES --------
    elif event_type == "preference":
        prefs = tool_context.state.get(
            "user.preferences",
            {
                "liked_movies": [],
                "disliked_movies": [],
                "favorite_genres": [],
                "preferred_moods": [],
            },
        )

        for key, values in payload.items():
            for v in values:
                if v not in prefs[key]:
                    prefs[key].append(v)

        tool_context.state["user.preferences"] = prefs

    return {"status": "memory_updated"}


# ==================================================
# AGENTS
# ==================================================

movie_list_agent = Agent(
    name="movie_list_agent",
    model="gemini-2.0-flash-lite",
    tools=[fetch_movies],
    output_key="movie.search.last",
    instruction="""
Search for movies using the user's query.

Rules:
- Call the movie search tool.
- Do NOT update memory.
- Return results strictly in MovieSearchResult format.
"""
)


movie_detail_agent = Agent(
    name="movie_detail_agent",
    model="gemini-2.0-flash-lite",
    tools=[google_search],
    output_key="movie.detail.last",
    instruction="""
Provide factual information about a movie.

Rules:
- Use google_search.
- Do NOT recommend movies.
- Do NOT update memory.
- Return results strictly in MovieDetail format.
"""
)


recommendation_agent = Agent(
    name="recommendation_agent",
    model="gemini-2.5-flash",
    output_key="movie.recommendations.last",
    instruction="""
Recommend EXACTLY 5 movies.

You may receive the following session state keys (any may be missing):
- user.preferences
- movie.search.history
- movie.recommendations.history

Rules:
- Treat missing keys as empty lists/dicts.
- If search history is empty, rely entirely on user.preferences.
- If preferences exist, they are sufficient to recommend.
- Never recommend a movie already present in movie.recommendations.history.
- NEVER respond with apologies, errors, or refusal messages.
- ALWAYS produce valid recommendations.

Output:
- Strictly follow MovieRecommendations schema.
"""
)


# ==================================================
# ROOT ORCHESTRATOR
# ==================================================

root_agent = LlmAgent(
    name="root_agent",
    model="gemini-2.5-flash",
    tools=[
        AgentTool(agent=movie_list_agent),
        AgentTool(agent=movie_detail_agent),
        AgentTool(agent=recommendation_agent),
        update_memory,
    ],
    instruction="""
You are the central orchestration agent.

Before routing:
- Ensure these state keys exist (initialize empty if missing):
  - user.preferences
  - movie.search.history
  - movie.recommendations.history

Responsibilities:
1. Detect user intent.
2. Route to the correct agent:
   - Search → movie_list_agent
   - Details → movie_detail_agent
   - Recommendations → recommendation_agent
3. After an agent completes:
   - Search → update_memory(event_type="search")
   - Recommendation → update_memory(event_type="recommendation")
   - Preference expression → update_memory(event_type="preference")

Rules:
- output_key values are ephemeral snapshots.
- update_memory is the ONLY persistence mechanism.
- Never invent preferences or history.
"""
)
