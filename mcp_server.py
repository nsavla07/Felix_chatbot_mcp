"""
MCP Server exposing Felix portfolio recommendation APIs and MongoDB chat tools.
Run with: python mcp_server.py
"""

from typing import Literal
from mcp.server.fastmcp import FastMCP
import requests
from pymongo import MongoClient
from datetime import datetime, timezone
from dotenv import load_dotenv
import uuid
import os

# Load .env so MONGO_URI is available
load_dotenv()

mcp = FastMCP("Felix Portfolio Advisor")

# Shared state — tokens and connection strings
# Reads MONGO_URI from .env file (your Atlas connection string)
_state = {
    "auth_token": os.getenv("AUTH_TOKEN", ""),
    "base_url": "https://felix.investica.com",
    "mongo_uri": os.getenv(
        "MONGO_URI",
        "mongodb://localhost:27017",
    ),
}

# ────────────────────────────────────────────
# MongoDB helper — reuses a single connection
# ────────────────────────────────────────────

_mongo_client = None


def _get_db():
    """
    Return the 'chatbot' database (matches your existing db.js setup).
    Creates the MongoClient once and reuses it.
    """
    print(">>> _get_db START")
    global _mongo_client
    if _mongo_client is None:
        print("  Connecting to MongoDB...")
        _mongo_client = MongoClient(_state["mongo_uri"], serverSelectionTimeoutMS=5000)
        _mongo_client.admin.command("ping")
        print("  MongoDB connected!")
    print("<<< _get_db END")
    return _mongo_client["chatbot"]


# ────────────────────────────────────────────
# Config tools
# ────────────────────────────────────────────


@mcp.tool()
def set_auth_token(token: str) -> str:
    """Set the Bearer token for authenticating with the Felix API.

    Args:
        token: The JWT Bearer token (without the 'Bearer ' prefix)
    """
    print(">>> set_auth_token START")
    _state["auth_token"] = token
    print("<<< set_auth_token END")
    return "Auth token updated."


@mcp.tool()
def set_mongo_uri(uri: str) -> str:
    """Set the MongoDB connection URI.

    Args:
        uri: Full MongoDB connection string (e.g. mongodb://localhost:27017)
    """
    print(">>> set_mongo_uri START")
    global _mongo_client
    _state["mongo_uri"] = uri
    _mongo_client = None          # force reconnect on next call
    print("<<< set_mongo_uri END")
    return "MongoDB URI updated."


# ────────────────────────────────────────────
# Chat session tools  (MongoDB-backed)
# ────────────────────────────────────────────


@mcp.tool()
def create_chat_session(user_name: str = "Guest") -> dict:
    """Create a new chat session in MongoDB. Returns the session_id.

    Args:
        user_name: Display name for the user
    """
    print(">>> create_chat_session START")
    db = _get_db()
    session_id = str(uuid.uuid4())
    db.chat_sessions.insert_one({
        "session_id": session_id,
        "user_name": user_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "messages": [],
    })
    print("<<< create_chat_session END")
    return {"session_id": session_id, "user_name": user_name}


@mcp.tool()
def save_chat_message(
    session_id: str,
    role: Literal["user", "assistant"],
    content: str,
) -> dict:
    """Append a message to a chat session in MongoDB.

    Args:
        session_id: The chat session ID
        role: Who sent the message — 'user' or 'assistant'
        content: The message text
    """
    print(">>> save_chat_message START")
    db = _get_db()
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    result = db.chat_sessions.update_one(
        {"session_id": session_id},
        {"$push": {"messages": message}},
    )
    if result.matched_count == 0:
        print("<<< save_chat_message END (session not found)")
        return {"error": "Session not found"}
    print("<<< save_chat_message END")
    return {"status": "saved"}


@mcp.tool()
def get_chat_history(session_id: str, limit: int = 50) -> dict:
    """Retrieve recent chat messages for a session.

    Args:
        session_id: The chat session ID
        limit: Max number of recent messages to return
    """
    print(">>> get_chat_history START")
    db = _get_db()
    session = db.chat_sessions.find_one(
        {"session_id": session_id}, {"_id": 0}
    )
    if not session:
        print("<<< get_chat_history END (session not found)")
        return {"error": "Session not found"}
    messages = session.get("messages", [])[-limit:]
    print("<<< get_chat_history END")
    return {
        "session_id": session_id,
        "user_name": session.get("user_name", ""),
        "messages": messages,
    }


@mcp.tool()
def list_chat_sessions(limit: int = 20) -> list:
    """List recent chat sessions stored in MongoDB.

    Args:
        limit: Max number of sessions to return
    """
    print(">>> list_chat_sessions START")
    db = _get_db()
    result = list(
        db.chat_sessions.find(
            {},
            {"_id": 0, "session_id": 1, "user_name": 1, "created_at": 1},
        )
        .sort("created_at", -1)
        .limit(limit)
    )
    print("<<< list_chat_sessions END")
    return result


@mcp.tool()
def rename_chat_session(session_id: str, name: str) -> dict:
    """Rename a chat session.

    Args:
        session_id: The chat session ID
        name: New display name for the session
    """
    print(">>> rename_chat_session START")
    db = _get_db()
    result = db.chat_sessions.update_one(
        {"session_id": session_id},
        {"$set": {"user_name": name}},
    )
    if result.matched_count == 0:
        print("<<< rename_chat_session END (session not found)")
        return {"error": "Session not found"}
    print("<<< rename_chat_session END")
    return {"status": "renamed", "name": name}


@mcp.tool()
def delete_chat_session(session_id: str) -> dict:
    """Delete a chat session from MongoDB.

    Args:
        session_id: The chat session ID to delete
    """
    print(">>> delete_chat_session START")
    db = _get_db()
    result = db.chat_sessions.delete_one({"session_id": session_id})
    if result.deleted_count == 0:
        print("<<< delete_chat_session END (session not found)")
        return {"error": "Session not found"}
    print("<<< delete_chat_session END")
    return {"status": "deleted"}


# ────────────────────────────────────────────
# Portfolio recommendation tools
# ────────────────────────────────────────────


def _felix_headers() -> dict:
    """Build common headers for Felix API calls."""
    print(">>> _felix_headers START")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if _state["auth_token"]:
        headers["Authorization"] = f"Bearer {_state['auth_token']}"
    print("<<< _felix_headers END")
    return headers


@mcp.tool()
def recommend_portfolio(
    client_name: str,
    age: int,
    investment_horizon_years: int,
    risk_appetite: Literal["Conservative", "Moderate", "Aggressive"],
    investment_type: Literal["Lumpsum", "SIP"],
    income_level: Literal["Below ₹10L", "₹10L-₹25L", "Above ₹25L"],
    employment_status: Literal["Salaried", "Self-employed", "Business Owner", "Retired"],
    financial_goals: Literal["Retirement", "Child Education", "Wealth Creation", "House Purchase", "Emergency Fund", "Tax Saving"],
    existing_investments: Literal["Under ₹10L", "₹10L-₹50L", "Above ₹50L"],
    home_ownership: Literal["Own", "Rent"],
    dependents: Literal["0", "1-2", "3-4", "5+"],
    tax_bracket: Literal["0-5%", "5-20%", "20-30%", "Above 30%"],
    investment_amount: float,
    auth_token: str = "",
    rate_of_return: float = 12.0,
    additional_notes: str = "",
    include_allocation: bool = True,
    free_think: bool = False,
) -> dict:
    """Generate an AI-recommended mutual fund portfolio based on client profile.

    Args:
        client_name: Full name of the client
        age: Client's age in years
        investment_horizon_years: How many years the client plans to stay invested
        risk_appetite: Risk tolerance level
        investment_type: Type of investment
        income_level: Annual income bracket
        employment_status: Employment type
        financial_goals: Primary financial goal
        existing_investments: Current investment value bracket
        home_ownership: Housing status
        dependents: Number of dependents
        tax_bracket: Tax bracket percentage
        investment_amount: Total amount to invest in INR
        auth_token: JWT Bearer token for Felix API authentication
        rate_of_return: Expected annual rate of return in percent
        additional_notes: Any extra context or preferences from the client
        include_allocation: Whether to include fund-wise allocation amounts
        free_think: Allow the AI to think freely beyond the template
    """
    print(">>> recommend_portfolio START")
    prompt = (
        f"Client Name: {client_name}\n"
        f"        Age: {age} years old\n"
        f"        Investment Horizon: {investment_horizon_years} years\n"
        f"        Risk Appetite: {risk_appetite}\n"
        f"        Investment Type: {investment_type}\n"
        f"        \n"
        f"        Income Level: {income_level}\n"
        f"        Employment Status: {employment_status}\n"
        f"        Financial Goals: {financial_goals}\n"
        f"        Existing Investments: {existing_investments}\n"
        f"        Home Ownership: {home_ownership}\n"
        f"        Dependents: {dependents}\n"
        f"        Tax Bracket: {tax_bracket}\n"
        f"        \n"
        f"        Additional Notes: {additional_notes}"
    )

    payload = {
        "prompt": prompt,
        "includeAllocation": include_allocation,
        "investmentAmount": investment_amount,
        "investmentType": investment_type.lower(),
        "freeThink": free_think,
        "rate_of_return": rate_of_return,
    }

    url = f"{_state['base_url']}/api/recommend-portfolio"
    headers = _felix_headers()
    token = auth_token or _state["auth_token"]
    if token:
        headers["Authorization"] = f"Bearer {token}"
    print(">>> recommend_portfolio calling API")
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
    except Exception as e:
        print("<<< recommend_portfolio END (request failed)")
        return {"error": f"Request failed: {e}"}
    print(f"<<< recommend_portfolio END (status {resp.status_code})")
    if resp.status_code != 200:
        return {"error": f"API returned {resp.status_code}", "detail": resp.text[:500]}
    return resp.json()


@mcp.tool()
def recommend_portfolio_goal_based(
    client_name: str,
    target_amount: float,
    years: int,
    risk_appetite: Literal["Conservative", "Moderate", "Aggressive"],
    investment_type: Literal["SIP", "Lumpsum"] = "SIP",
    auth_token: str = "",
    rate_of_return: float = 12.0,
    apply_inflation: bool = True,
    prompt: str = "",
) -> dict:
    """Generate a goal-based mutual fund portfolio recommendation.

    Args:
        client_name: Full name of the client
        target_amount: The target corpus amount in INR
        years: Number of years to achieve the goal
        risk_appetite: Risk tolerance level
        investment_type: Monthly SIP or one-time Lumpsum
        auth_token: JWT Bearer token for Felix API authentication
        rate_of_return: Expected annual return percentage
        apply_inflation: Whether to adjust the target for inflation
        prompt: Additional context or goal description
    """
    print(">>> recommend_portfolio_goal_based START")
    payload = {
        "name": client_name,
        "target_amount": target_amount,
        "years": years,
        "risk_appetite": risk_appetite,
        "investment_type": investment_type.lower(),
        "rate_of_return": rate_of_return,
        "apply_inflation": apply_inflation,
        "prompt": prompt,
    }

    url = f"{_state['base_url']}/api/recommend-portfolio-goal-based"
    headers = _felix_headers()
    token = auth_token or _state["auth_token"]
    if token:
        headers["Authorization"] = f"Bearer {token}"
    print(">>> recommend_portfolio_goal_based calling API")
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
    except Exception as e:
        print("<<< recommend_portfolio_goal_based END (request failed)")
        return {"error": f"Request failed: {e}"}
    print(f"<<< recommend_portfolio_goal_based END (status {resp.status_code})")
    if resp.status_code != 200:
        return {"error": f"API returned {resp.status_code}", "detail": resp.text[:500]}
    return resp.json()


@mcp.tool()
def get_tool_schemas() -> list[dict]:
    """Return the parameter schemas for all portfolio tools."""
    print(">>> get_tool_schemas START")
    tools_info = []
    for tool_name, tool_obj in mcp._tool_manager._tools.items():
        if tool_name == "get_tool_schemas":
            continue
        tools_info.append({
            "name": tool_obj.name,
            "description": tool_obj.description,
            "parameters": tool_obj.parameters,
        })
    print("<<< get_tool_schemas END")
    return tools_info


if __name__ == "__main__":
    mcp.run()
