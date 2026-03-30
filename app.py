"""
Streamlit Chat UI for Felix Portfolio Advisor.

Uses:
  - OpenAI            → conversational AI (gathers info naturally, calls tools)
  - MCP tools         → portfolio recommendations via Felix API
  - MongoDB           → persistent chat history across sessions

Run with: streamlit run app.py
"""

import streamlit as st
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from mcp_server import (
    _state,
    recommend_portfolio,
    recommend_portfolio_goal_based,
    create_chat_session,
    save_chat_message,
    get_chat_history,
    list_chat_sessions,
    rename_chat_session,
    delete_chat_session,
)

# ────────────────────────────────────────────
# System prompt — tells Gemini HOW to converse
# ────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are Felix, a friendly and knowledgeable mutual fund portfolio advisor.
You help users find the right investment portfolio through natural conversation.

Available tools:
1. recommend_portfolio — comprehensive AI portfolio based on full client profile
2. recommend_portfolio_goal_based — goal-oriented portfolio (target amount + timeline)

Conversation style:
- Be warm, professional, and approachable.
- Ask 1–2 questions at a time — never dump a long form on the user.
- If the user is unsure about something (e.g. risk appetite), explain the options simply.
- When you have enough info, call the right tool.
- After receiving the tool result, summarise the recommendation in plain language.
- You may answer general investment / mutual-fund questions even without calling a tool.
- Format currency amounts with the ₹ symbol and commas (e.g. ₹5,00,000).
"""

# ────────────────────────────────────────────
# Tool definitions for Gemini function calling
# ────────────────────────────────────────────

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recommend_portfolio",
            "description": (
                "Generate an AI-recommended mutual fund portfolio. "
                "Call this when you have gathered enough client details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "client_name": {"type": "string", "description": "Full name"},
                    "age": {"type": "integer", "description": "Age in years"},
                    "investment_horizon_years": {"type": "integer", "description": "Years to stay invested"},
                    "risk_appetite": {"type": "string", "enum": ["Conservative", "Moderate", "Aggressive"]},
                    "investment_type": {"type": "string", "enum": ["Lumpsum", "SIP"]},
                    "income_level": {"type": "string", "enum": ["Below ₹10L", "₹10L-₹25L", "Above ₹25L"]},
                    "employment_status": {"type": "string", "enum": ["Salaried", "Self-employed", "Business Owner", "Retired"]},
                    "financial_goals": {"type": "string", "enum": ["Retirement", "Child Education", "Wealth Creation", "House Purchase", "Emergency Fund", "Tax Saving"]},
                    "existing_investments": {"type": "string", "enum": ["Under ₹10L", "₹10L-₹50L", "Above ₹50L"]},
                    "home_ownership": {"type": "string", "enum": ["Own", "Rent"]},
                    "dependents": {"type": "string", "enum": ["0", "1-2", "3-4", "5+"]},
                    "tax_bracket": {"type": "string", "enum": ["0-5%", "5-20%", "20-30%", "Above 30%"]},
                    "investment_amount": {"type": "number", "description": "Amount in INR"},
                    "rate_of_return": {"type": "number", "description": "Expected annual return, default 12.0"},
                    "additional_notes": {"type": "string", "description": "Extra context"},
                    "include_allocation": {"type": "boolean", "description": "Include fund-wise amounts"},
                    "free_think": {"type": "boolean", "description": "Allow free thinking"},
                },
                "required": [
                    "client_name", "age", "investment_horizon_years", "risk_appetite",
                    "investment_type", "income_level", "employment_status",
                    "financial_goals", "existing_investments", "home_ownership",
                    "dependents", "tax_bracket", "investment_amount",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_portfolio_goal_based",
            "description": (
                "Generate a goal-based mutual fund portfolio. "
                "Use when the user has a specific target amount and timeline."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "client_name": {"type": "string", "description": "Full name"},
                    "target_amount": {"type": "number", "description": "Target corpus in INR"},
                    "years": {"type": "integer", "description": "Years to reach the goal"},
                    "risk_appetite": {"type": "string", "enum": ["Conservative", "Moderate", "Aggressive"]},
                    "investment_type": {"type": "string", "enum": ["SIP", "Lumpsum"]},
                    "rate_of_return": {"type": "number", "description": "Expected annual return, default 12.0"},
                    "apply_inflation": {"type": "boolean", "description": "Adjust for inflation"},
                    "prompt": {"type": "string", "description": "Additional context"},
                },
                "required": ["client_name", "target_amount", "years", "risk_appetite"],
            },
        },
    },
]

# Map tool names → actual Python functions from the MCP server
TOOL_FUNCTIONS = {
    "recommend_portfolio": recommend_portfolio,
    "recommend_portfolio_goal_based": recommend_portfolio_goal_based,
}

# ────────────────────────────────────────────
# Session-state helpers
# ────────────────────────────────────────────

WELCOME_MESSAGE = (
    "Hi! I'm **Felix**, your portfolio advisor. "
    "Tell me a bit about yourself and your investment goals, "
    "and I'll recommend the right mutual fund portfolio for you.\n\n"
    "You can say something like:\n"
    "- *I want to invest ₹50,000 per month via SIP*\n"
    "- *I need ₹1 crore in 10 years for my child's education*\n"
    "- *What portfolio suits a conservative 45-year-old?*"
)


def init_session():
    """Set up Streamlit session state defaults."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    # openai_history stores the OpenAI message dicts for multi-turn chat
    if "openai_history" not in st.session_state:
        st.session_state.openai_history = []


def start_new_session():
    """Create a fresh MongoDB session and reset local state."""
    try:
        result = create_chat_session()
        st.session_state.session_id = result["session_id"]
    except Exception:
        st.session_state.session_id = None   # offline mode
    st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
    st.session_state.openai_history = []
    # Save the welcome message to MongoDB
    if st.session_state.session_id:
        try:
            save_chat_message(st.session_state.session_id, "assistant", WELCOME_MESSAGE)
        except Exception:
            pass


def load_session(session_id: str):
    """Load an existing session from MongoDB."""
    result = get_chat_history(session_id)
    if "error" in result:
        st.error(f"Could not load session: {result['error']}")
        return
    st.session_state.session_id = session_id
    st.session_state.messages = [
        {"role": m["role"], "content": m["content"]}
        for m in result.get("messages", [])
    ]
    # Rebuild OpenAI history from display messages.
    st.session_state.openai_history = []
    for m in st.session_state.messages:
        st.session_state.openai_history.append(
            {"role": m["role"], "content": m["content"]}
        )


def persist_message(role: str, content: str):
    """Save a message to both local state and MongoDB."""
    st.session_state.messages.append({"role": role, "content": content})
    if st.session_state.session_id:
        try:
            save_chat_message(st.session_state.session_id, role, content)
        except Exception:
            pass


# ────────────────────────────────────────────
# Gemini AI — conversation + tool-use loop
# ────────────────────────────────────────────


def get_ai_response(user_message: str):
    """
    Send the conversation to OpenAI, handle any tool calls (function calling),
    and return the final text reply.
    """
    client = OpenAI(api_key=st.session_state.openai_key)

    # Add user message to history
    st.session_state.openai_history.append({"role": "user", "content": user_message})

    # Build messages with system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.openai_history

    # Call OpenAI with tools
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=OPENAI_TOOLS,
    )

    message = response.choices[0].message

    # Tool-use loop: OpenAI may request function calls.
    while message.tool_calls:
        # Add assistant message (with tool_calls) to history
        st.session_state.openai_history.append(message.model_dump())

        # Execute each tool call
        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            # Cast types for int fields
            if fn_name == "recommend_portfolio":
                if "age" in fn_args:
                    fn_args["age"] = int(fn_args["age"])
                if "investment_horizon_years" in fn_args:
                    fn_args["investment_horizon_years"] = int(fn_args["investment_horizon_years"])
            elif fn_name == "recommend_portfolio_goal_based":
                if "years" in fn_args:
                    fn_args["years"] = int(fn_args["years"])

            # Call the MCP tool function
            try:
                result = TOOL_FUNCTIONS[fn_name](**fn_args)
                result_data = result if isinstance(result, dict) else {"result": str(result)}
            except Exception as exc:
                result_data = {"error": str(exc)}

            # Add tool result to history
            st.session_state.openai_history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result_data),
            })

        # Get next response
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.openai_history
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=OPENAI_TOOLS,
        )
        message = response.choices[0].message

    # Add final assistant response to history
    reply = message.content or ""
    st.session_state.openai_history.append({"role": "assistant", "content": reply})

    return reply


# ────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────


def main():
    st.set_page_config(
        page_title="Felix Portfolio Advisor",
        page_icon="💬",
        layout="wide",
    )
    init_session()

    # ── Sidebar ──────────────────────────────
    with st.sidebar:
        st.title("Felix Advisor")

        # Load API key and MongoDB URI from Streamlit Secrets (cloud) or .env (local)
        def _get_secret(key, default=""):
            try:
                return st.secrets[key]
            except (KeyError, FileNotFoundError):
                return os.getenv(key, default)

        st.session_state.openai_key = _get_secret("OPENAI_API_KEY")
        _state["mongo_uri"] = _get_secret("MONGO_URI", "mongodb://localhost:27017")

        # Bearer token — user must enter manually
        st.markdown("##### Configuration")
        token = st.text_input(
            "Felix Bearer Token", type="password",
            help="JWT token from felix.investica.com",
        )
        if token:
            _state["auth_token"] = token
            st.success("Felix token set", icon="✅")

        # Session management
        st.markdown("---")
        st.markdown("##### Chat Sessions")

        if st.button("+ New Chat", use_container_width=True):
            start_new_session()
            st.rerun()

        # Rename dialog (show above the list so it doesn't vanish)
        if st.session_state.get("renaming_session"):
            rename_sid = st.session_state.renaming_session
            new_name = st.text_input("New name:", key="rename_input")
            r1, r2 = st.columns(2)
            with r1:
                if st.button("Save", use_container_width=True):
                    if new_name.strip():
                        rename_chat_session(rename_sid, new_name.strip())
                        del st.session_state.renaming_session
                        st.rerun()
            with r2:
                if st.button("Cancel", use_container_width=True):
                    del st.session_state.renaming_session
                    st.rerun()
            st.markdown("---")

        # Handle pending actions from previous run
        if st.session_state.get("pending_load"):
            load_session(st.session_state.pending_load)
            del st.session_state.pending_load
            st.rerun()

        if st.session_state.get("pending_delete"):
            del_sid = st.session_state.pending_delete
            delete_chat_session(del_sid)
            if st.session_state.session_id == del_sid:
                start_new_session()
            del st.session_state.pending_delete
            st.rerun()

        # Session list
        try:
            sessions = list_chat_sessions(limit=10)
        except Exception:
            sessions = []
            st.caption("MongoDB not connected — history unavailable")

        for s in sessions:
            sid = s.get("session_id", "")
            label = s.get("user_name", "Guest")
            created = s.get("created_at", "")[:10]
            col1, col2, col3 = st.columns([4, 1, 1])
            with col1:
                if st.button(
                    f"{label} — {created}",
                    key=f"load_{sid}",
                    use_container_width=True,
                ):
                    st.session_state.pending_load = sid
                    st.rerun()
            with col2:
                if st.button("✏️", key=f"ren_{sid}", help="Rename"):
                    st.session_state.renaming_session = sid
                    st.rerun()
            with col3:
                if st.button("🗑", key=f"del_{sid}", help="Delete"):
                    st.session_state.pending_delete = sid
                    st.rerun()

    # ── Main area ────────────────────────────
    st.title("💬 Felix Portfolio Advisor")
    st.caption(
        "Chat naturally about your goals — "
        "Felix will recommend the right portfolio when ready"
    )

    # Auto-create first session
    if not st.session_state.messages:
        start_new_session()

    # Display conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Tell me about your investment goals..."):
        # Guard: need API keys
        if not st.session_state.openai_key:
            st.warning("Enter your **OpenAI API key** in the sidebar to start chatting.")
            st.stop()
        if not _state["auth_token"]:
            st.warning("Enter your **Felix Bearer token** in the sidebar.")
            st.stop()

        # Show & save user message
        with st.chat_message("user"):
            st.markdown(prompt)
        persist_message("user", prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Felix is thinking..."):
                try:
                    reply = get_ai_response(prompt)
                    st.markdown(reply)
                    persist_message("assistant", reply)
                except Exception as exc:
                    st.error(f"Something went wrong: {exc}")


if __name__ == "__main__":
    main()
