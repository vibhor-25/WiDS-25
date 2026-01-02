# 🎬 Movie Recommendation Agent (Google ADK)

This directory contains a **stateful, multi-agent movie assistant** built using **Google Agent Development Kit (ADK)**.

The agent can:
- 🔍 Search for movies
- 📖 Provide factual movie details
- 🎯 Recommend personalized movies based on user preferences and mood

It uses **agent orchestration**, **tool calling**, and **session state tracking**.

---

## 🧠 Agent Architecture

### 1️⃣ Root Agent
- Identifies user intent (search / detail / recommend)
- Maintains session state (preferences & history)
- Routes requests to sub-agents

### 2️⃣ Movie Listing Agent
- Uses OMDb API
- Lists movies matching a query
- No recommendations or ranking

### 3️⃣ Movie Detail Agent
- Uses Google Search tool
- Provides factual information only

### 4️⃣ Recommendation Agent
- Generates EXACTLY 5 movie recommendations
- Uses preferences, mood, and history
- Avoids repetition

---

## 📁 Directory Structure

```
movie_bot/
├── __init__.py
├── agent.py
└── README.md
```

---

## 🔐 Environment Setup

Create a `.env` file in the project root:

```env
MOVIE_API_KEY=your_omdb_api_key
GOOGLE_API_KEY=your_google_api_key
GOOGLE_GENAI_USE_VERTEXAI=0
```

---

## 📦 Installation

```bash
python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate   # Linux / Mac

pip install google-adk python-dotenv requests
```



---

## ▶️ Run the Agent

From repo root:

```bash
adk web --port 8000
```

Open:
```
http://localhost:8000
```

---

## 🧪 Example Prompts

- Search movies titled - "Batman"
- Who directed Interstellar?
- Recommend movies for a sad mood
- I like sci-fi movies
