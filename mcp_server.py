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
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("felix-mcp")

# Load .env so MONGO_URI is available
load_dotenv()

mcp = FastMCP("Felix Portfolio Advisor")

# Shared state — tokens and connection strings
# Reads MONGO_URI from .env file (your Atlas connection string)
_state = {
    "auth_token": "",
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
    global _mongo_client
    if _mongo_client is None:
        log.info("[MongoDB] Connecting to: %s", _state["mongo_uri"][:40] + "...")
        try:
            _mongo_client = MongoClient(_state["mongo_uri"], serverSelectionTimeoutMS=5000)
            # Test the connection
            _mongo_client.admin.command("ping")
            log.info("[MongoDB] Connection successful")
        except Exception as exc:
            log.error("[MongoDB] Connection FAILED: %s", exc)
            raise
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
    _state["auth_token"] = token
    return "Auth token updated."


@mcp.tool()
def set_mongo_uri(uri: str) -> str:
    """Set the MongoDB connection URI.

    Args:
        uri: Full MongoDB connection string (e.g. mongodb://localhost:27017)
    """
    global _mongo_client
    _state["mongo_uri"] = uri
    _mongo_client = None          # force reconnect on next call
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
    log.info("[create_chat_session] Creating session for user: %s", user_name)
    db = _get_db()
    session_id = str(uuid.uuid4())
    db.chat_sessions.insert_one({
        "session_id": session_id,
        "user_name": user_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "messages": [],
    })
    log.info("[create_chat_session] Created session: %s", session_id)
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
    log.info("[save_chat_message] Saving %s message to session %s (%d chars)", role, session_id[:8], len(content))
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
        log.warning("[save_chat_message] Session %s not found", session_id[:8])
        return {"error": "Session not found"}
    log.info("[save_chat_message] Saved successfully")
    return {"status": "saved"}


@mcp.tool()
def get_chat_history(session_id: str, limit: int = 50) -> dict:
    """Retrieve recent chat messages for a session.

    Args:
        session_id: The chat session ID
        limit: Max number of recent messages to return
    """
    db = _get_db()
    session = db.chat_sessions.find_one(
        {"session_id": session_id}, {"_id": 0}
    )
    if not session:
        return {"error": "Session not found"}
    messages = session.get("messages", [])[-limit:]
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
    db = _get_db()
    return list(
        db.chat_sessions.find(
            {},
            {"_id": 0, "session_id": 1, "user_name": 1, "created_at": 1},
        )
        .sort("created_at", -1)
        .limit(limit)
    )


@mcp.tool()
def rename_chat_session(session_id: str, name: str) -> dict:
    """Rename a chat session.

    Args:
        session_id: The chat session ID
        name: New display name for the session
    """
    db = _get_db()
    result = db.chat_sessions.update_one(
        {"session_id": session_id},
        {"$set": {"user_name": name}},
    )
    if result.matched_count == 0:
        return {"error": "Session not found"}
    return {"status": "renamed", "name": name}


@mcp.tool()
def delete_chat_session(session_id: str) -> dict:
    """Delete a chat session from MongoDB.

    Args:
        session_id: The chat session ID to delete
    """
    db = _get_db()
    result = db.chat_sessions.delete_one({"session_id": session_id})
    if result.deleted_count == 0:
        return {"error": "Session not found"}
    return {"status": "deleted"}


# ────────────────────────────────────────────
# Portfolio recommendation tools
# ────────────────────────────────────────────


def _felix_headers() -> dict:
    """Build common headers for Felix API calls."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if _state["auth_token"]:
        headers["Authorization"] = f"Bearer {_state['auth_token']}"
        log.info("[_felix_headers] Auth token set (length=%d)", len(_state["auth_token"]))
    else:
        log.warning("[_felix_headers] No auth token set!")
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
        rate_of_return: Expected annual rate of return in percent
        additional_notes: Any extra context or preferences from the client
        include_allocation: Whether to include fund-wise allocation amounts
        free_think: Allow the AI to think freely beyond the template
    """
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
    log.info("[recommend_portfolio] POST %s — client=%s, amount=%.0f, type=%s", url, client_name, investment_amount, investment_type)
    log.info("[recommend_portfolio] Payload: %s", json.dumps(payload, default=str)[:300])
    resp = requests.post(url, json=payload, headers=_felix_headers(), timeout=120)
    log.info("[recommend_portfolio] Response status: %d", resp.status_code)
    if resp.status_code != 200:
        log.error("[recommend_portfolio] Response body: %s", resp.text[:500])
    resp.raise_for_status()
    data = resp.json()
    log.info("[recommend_portfolio] SUCCESS — response keys: %s", list(data.keys()) if isinstance(data, dict) else "not a dict")
    return data


@mcp.tool()
def recommend_portfolio_goal_based(
    client_name: str,
    target_amount: float,
    years: int,
    risk_appetite: Literal["Conservative", "Moderate", "Aggressive"],
    investment_type: Literal["SIP", "Lumpsum"] = "SIP",
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
        rate_of_return: Expected annual return percentage
        apply_inflation: Whether to adjust the target for inflation
        prompt: Additional context or goal description
    """
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
    log.info("[recommend_portfolio_goal_based] POST %s — client=%s, target=%.0f, years=%d", url, client_name, target_amount, years)
    log.info("[recommend_portfolio_goal_based] Payload: %s", json.dumps(payload, default=str)[:300])
    resp = requests.post(url, json=payload, headers=_felix_headers(), timeout=120)
    log.info("[recommend_portfolio_goal_based] Response status: %d", resp.status_code)
    if resp.status_code != 200:
        log.error("[recommend_portfolio_goal_based] Response body: %s", resp.text[:500])
    resp.raise_for_status()
    data = resp.json()
    log.info("[recommend_portfolio_goal_based] SUCCESS — response keys: %s", list(data.keys()) if isinstance(data, dict) else "not a dict")
    return data


@mcp.tool()
def get_tool_schemas() -> list[dict]:
    """Return the parameter schemas for all portfolio tools."""
    tools_info = []
    for tool_name, tool_obj in mcp._tool_manager._tools.items():
        if tool_name == "get_tool_schemas":
            continue
        tools_info.append({
            "name": tool_obj.name,
            "description": tool_obj.description,
            "parameters": tool_obj.parameters,
        })
    return tools_info


if __name__ == "__main__":
    mcp.run()
