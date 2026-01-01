# from asyncio import Runner
from google.adk.agents.llm_agent import Agent
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool
# from google.adk.session.in_memory_session_service import InMemorySessionService

import os
from dotenv import load_dotenv
import requests

load_dotenv()
api_key = os.getenv("MOVIE_API_KEY")

# os.environ["GOOGLE_API_KEY"] =  os.getenv("GOOGLE_API_KEY")
# os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = os.getenv("GOOGLE_GENAI_USE_VERTEXAI")

URL = f"https://www.omdbapi.com/?apikey={api_key}&s="

# Define constants for identifying the interaction context
# USER_ID = "user_1"
# SESSION_ID = "session_001"  # Using a fixed ID for simplicity

# State Schema
state = {
    "preferences": {
        "genres": [],
        "moods": [],
        "liked_movies": [],
        "disliked_movies": [],
    },
    "history": {"searched_movies": [], "recommended_movies": []},
}


# session_service = InMemorySessionService()

# runner = Runner(
#     agent=root_agent,  # The agent we want to run
#     session_service=session_service,  # Uses our session manager
# )


def get_movie_info(query: str) -> dict:
    """Retrieves a list of movies and their description for a specified movie query.

    Args:
        title (str): The title of the movie queried by the user(e.g., "Inception", "The Matrix").

    Returns:
        dict: A dictionary containing the movie information.
              Includes a 'status' key ('success' or 'error').
              If 'success', includes a 'report' key with list of matched movies and details, Send this to LLM with clear instructions to format it nicely excluding the imdb id.
              If 'error', includes an 'error_message' key.
    """
    print(
        f"--- Tool: get_movie_info called for list of movies: {query} ---"
    )  # Log tool execution
    print(state)  # Log current state
    response = requests.get(URL + query)
    data = response.json()
    if data["Response"] == "True":
        return {"status": "success", "report": data["Search"]}
    else:
        return {
            "status": "error",
            "error_message": f"Sorry, I couldn't find any information for '{query}'.",
        }


# Recommendation Agent
recommendation_agent = Agent(
    model="gemini-2.5-flash",
    name="recommendation_agent",
    description="A reasoning-based movie recommendation agent that suggests movies "
    "using user preferences, mood, and conversation history. ",
    instruction="""
You are an expert movie recommendation assistant.

You receive:
- User preferences (genres, moods)
- Liked and disliked movies
- Conversation history

Your task:
- Recommend EXACTLY 5 movies
- Each recommendation must include:
    - Movie title
    - One-line explanation of why it fits the user's taste

Rules:
- Prefer relevance over popularity
- Avoid repeating previously recommended movies
- If preferences are vague, ask a clarifying question
- If mood is given, prioritize mood over genre
""",
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

When the user asks:
- Who directed a movie
- Cast information
- Plot, awards, or background

Steps:
1. Use google_search
2. Extract reliable information
3. Summarize concisely

Rules:
- Do not speculate
- Do not recommend movies
- Cite commonly known facts only
""",
    tools=[google_search],
)

# Listing Agent
movie_list_agent = Agent(
    model="gemini-2.0-flash-lite",
    name="movie_list_agent",
    description="Searches and lists movies based on a user query using an external movie API. "
    "Used strictly for discovery, not recommendations.",
    instruction="""
You are a movie search assistant.

When the user:
- Asks to search movies
- Mentions a title but wants options

Steps:
1. Call the get_movie_info tool
2. Present results clearly:
   - Title
   - Year
   - Type (movie/series)

Rules:
- Do not recommend or rank movies
- Do not infer preferences
- If multiple movies share a name, list all
""",
    tools=[get_movie_info],
)


# Convert the specialized agents into callable tools for the main coordinator
detail_tool = AgentTool(agent=movie_detail_agent)
list_tool = AgentTool(agent=movie_list_agent)
recommend_tool = AgentTool(agent=recommendation_agent)

# Main agent
instruction = """
You are a stateful movie assistant responsible for orchestration.

Your responsibilities:
1. Identify user intent:
   - search → movie_list_agent
   - detail → movie_detail_agent
   - recommend → recommendation_agent

2. Extract and store preferences:
   - genres
   - moods
   - liked / disliked movies

3. Update session state accordingly.
4. Each new session starts with empty preferences and history.
5. Whenever you infer user preferences (genre, mood, liked or disliked movies), explicitly store them in the session state under:
-state.preferences.genres
-state.preferences.moods
-state.preferences.liked_movies
-state.preferences.disliked_movies
Also update:
-state.history.searched_movies
-state.history.recommended_movies


Routing rules:
- If the user asks for recommendations:
    - Call recommendation_agent
    - Provide it with session state
- If user mentions a genre or mood:
    - Store it in preferences
- If user says "like X":
    - Add X to liked_movies
- If intent is ambiguous:
    - Ask a clarification question
"""

root_agent = Agent(
    model="gemini-2.5-flash",
    name="root_agent",
    description="You are the main movie assistant agent that coordinates between the list and detail agents.",
    instruction=instruction,
    # sub_agents=[movie_list_agent, movie_detail_agent], # -> Won't work
    tools=[list_tool, detail_tool, recommend_tool],
)
