# from asyncio import Runner
from google.adk.agents import LlmAgent,Agent
from google.adk.tools import google_search
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.tools import AgentTool
from google.adk.tools import ToolContext

from .movie.movie_model import (
    MovieSearchItem,
    MovieSearchResult,
    MovieRecommendation,
    MovieRecommendations,
    MovieDetail,
)
import os
from dotenv import load_dotenv
import requests

load_dotenv()
api_key = os.getenv("MOVIE_API_KEY")

# os.environ["GOOGLE_API_KEY"] =  os.getenv("GOOGLE_API_KEY")
# os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = os.getenv("GOOGLE_GENAI_USE_VERTEXAI")

URL = f"https://www.omdbapi.com/?apikey={api_key}&s="

# State Schema
# Note: Session state is managed by the ADK session service. Keep this as a local default
# for quick testing but do not rely on this global as the authoritative session state.
state = {
    "preferences": {
        "genres": [],
        "moods": [],
        "liked_movies": [],
        "disliked_movies": [],
    },
    "history": {"searched_movies": [], "recommended_movies": []},
}


def get_movie_info(query: str, tool_context: ToolContext) -> dict:
    print(f"--- Tool: get_movie_info called for query: {query} ---")

    response = requests.get(URL + query)
    data = response.json()

    # ---- STATE UPDATE (VISIBLE IN UI) ----
    history = tool_context.state.get("history", {})
    searched = history.get("searched_movies", [])
    if query not in searched:
        searched.append(query)

    tool_context.state["history"] = {
        **history,
        "searched_movies": searched,
    }

    if data.get("Response") == "True":
        return {
            "query": query,
            "results": [
                {
                    "title": m.get("Title"),
                    "year": m.get("Year"),
                    "type": m.get("Type"),
                }
                for m in data.get("Search", [])
            ],
        }

    return {"query": query, "results": []}

# Recommendation Agent
recommendation_agent = Agent(
    model="gemini-2.5-flash",
    name="recommendation_agent",
    description="A reasoning-based movie recommendation agent that suggests movies "
    "using user preferences, mood, and conversation history. ",
    instruction="""
You are an expert movie recommendation assistant.

You operate using the session state provided to you.

The session state has the following structure:

state = {
  "preferences": {
    "genres": list[str],
    "moods": list[str],
    "liked_movies": list[str],
    "disliked_movies": list[str]
  },
  "history": {
    "searched_movies": list[str],
    "recommended_movies": list[str]
    }
}

You MUST read from:
- state.preferences.genres
- state.preferences.moods
- state.preferences.liked_movies
- state.preferences.disliked_movies
- state.history.recommended_movies

Your task:
- Recommend EXACTLY 5 movies
- Each recommendation must include:
  - Movie title
  - One concise line explaining why it matches the user's preferences or mood

Rules:
- Prefer relevance to state.preferences over popularity
- NEVER recommend a movie already present in state.history.recommended_movies
- If state.preferences.moods is non-empty, prioritize mood over genre
- If state.preferences is empty or vague, ask a clarifying question instead of guessing
- Do NOT modify the state directly
- Return results using MovieRecommendations schema.

""",
    output_key="movie_recommendations",

)

# Detailing Agent
movie_detail_agent = Agent(
    model="gemini-2.0-flash-lite",
    name="movie_detail_agent",
    description=(
        "A stateful movie assistant that orchestrates search, factual lookup, "
        "and personalized movie recommendations. It interprets user intent, "
        "tracks preferences such as genres, moods, and liked movies across the "
        "conversation, and delegates tasks to specialized agents when needed."
    ),
    instruction="""
You are a factual movie information assistant.

You receive the session state for context only.

State schema:

state = {
  "preferences": {...},
  "history": {
    "searched_movies": list[str],
    "recommended_movies": list[str]
  }
}

Your role:
- Answer factual questions only, such as:
  - Director
  - Cast
  - Plot summary
  - Awards
  - Background information

Steps:
1. Use google_search to retrieve reliable information
2. Extract verifiable facts
3. Respond with a concise, neutral summary

Rules:
- Do NOT recommend movies
- Do NOT infer or update preferences
- Do NOT speculate
- Do NOT modify state
- Use state only for conversational awareness
- Return details strictly using MovieDetail schema.
""",
    tools=[google_search],
    output_key="last_movie_detail",

)

# Listing Agent
movie_list_agent = Agent(
    model="gemini-2.0-flash-lite",
    name="movie_list_agent",
    description="Searches and lists movies based on a user query using an external movie API. "
    "Used strictly for discovery, not recommendations.",
    instruction="""
You are a movie discovery and listing assistant.

You operate with access to the session state for awareness only.

State schema:

state = {
  "preferences": {...},
  "history": {
    "searched_movies": list[str]
  }
}

When the user intent is search or discovery:
1. Call the get_movie_info tool with the user query
2. Present results clearly with:
   - Title
   - Year
   - Type (movie or series)

Rules:
- Do NOT recommend or rank movies
- Do NOT infer preferences
- Do NOT update state directly
- If multiple movies share the same title, list all results clearly
- Format output strictly according to MovieSearchResult schema.
""",
    tools=[get_movie_info],
    output_key="last_search_result",
    

)


# Sub-agents are registered directly on the root agent (no AgentTool wrappers needed).

# Main agent
instruction = """
You are a stateful movie assistant responsible for orchestration and state management.

You control and update the session state.

The session state schema is:

state = {
  "preferences": {
    "genres": list[str],
    "moods": list[str],
    "liked_movies": list[str],
    "disliked_movies": list[str]
  },
  "history": {
    "searched_movies": list[str],
    "recommended_movies": list[str]
  }
}

Your responsibilities:

1. Identify user intent:
   - Search or discovery → call movie_list_agent
   - Factual details → call movie_detail_agent
   - Recommendations → call recommendation_agent

2. State updates (MANDATORY when inferred):
   - If a genre is mentioned → append to state.preferences.genres
   - If a mood is mentioned → append to state.preferences.moods
   - If the user likes a movie → append to state.preferences.liked_movies
   - If the user dislikes a movie → append to state.preferences.disliked_movies
   - If a movie is searched → append to state.history.searched_movies
   - If movies are recommended → append titles to state.history.recommended_movies

3. State rules:
   - Each new session starts with empty lists
   - Never overwrite lists; always append
   - Avoid duplicates where possible

4. Routing rules:
   - For recommendations, always pass the full state to recommendation_agent
   - If intent is ambiguous, ask a clarification question before acting

You are the ONLY agent allowed to modify session state.
"""


list_tool = AgentTool(agent=movie_list_agent)
detail_tool = AgentTool(agent=movie_detail_agent)
recommend_tool = AgentTool(agent=recommendation_agent)
root_agent = LlmAgent(
    model="gemini-2.5-flash",
    name="root_agent",
    description="You are the main movie assistant agent that coordinates between the list, detail, recommendation agents.",
    instruction=instruction,
    tools=[list_tool, detail_tool, recommend_tool],
)
