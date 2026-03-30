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
import logging
from dotenv import load_dotenv
from openai import OpenAI

# ────────────────────────────────────────────
# Logger setup
# ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("felix-app")

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
# System prompt
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
- You may answer general investment / mutual-fund questions even without calling a tool.
- Format currency amounts with the ₹ symbol and Indian commas (e.g. ₹5,00,000).

After receiving a tool result, present the recommendation clearly using this format:
- Start with a brief summary of the user's profile/goal.
- Show the recommended portfolio in a clean **table** (Fund Name, Category, Allocation %).
- If amounts are included, show the fund-wise ₹ breakdown.
- Add a short note on why this allocation suits the user.
- End with a disclaimer: "This is an AI-generated suggestion. Please consult a certified financial advisor before investing."

If the tool returns an error, tell the user clearly what went wrong and ask them to try again.
"""

# ────────────────────────────────────────────
# Tool definitions for OpenAI function calling
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
    log.info("[init_session] Initializing session state")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "openai_history" not in st.session_state:
        st.session_state.openai_history = []
    log.info("[init_session] Done — session_id=%s, messages=%d", st.session_state.session_id, len(st.session_state.messages))


def start_new_session():
    """Create a fresh MongoDB session and reset local state."""
    log.info("[start_new_session] Creating new session...")
    try:
        result = create_chat_session()
        st.session_state.session_id = result["session_id"]
        log.info("[start_new_session] MongoDB session created: %s", st.session_state.session_id)
    except Exception as exc:
        st.session_state.session_id = None
        log.error("[start_new_session] MongoDB session creation FAILED: %s", exc)
    st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
    st.session_state.openai_history = []
    if st.session_state.session_id:
        try:
            save_chat_message(st.session_state.session_id, "assistant", WELCOME_MESSAGE)
            log.info("[start_new_session] Welcome message saved to MongoDB")
        except Exception as exc:
            log.error("[start_new_session] Failed to save welcome message: %s", exc)


def auto_rename_session(user_message: str):
    """Auto-rename the session based on the first user message."""
    if not st.session_state.session_id:
        return
    user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
    if len(user_msgs) > 0:
        return
    label = user_message.strip()[:40]
    if len(user_message.strip()) > 40:
        label += "..."
    log.info("[auto_rename_session] Renaming session %s to '%s'", st.session_state.session_id, label)
    try:
        rename_chat_session(st.session_state.session_id, label)
        log.info("[auto_rename_session] Rename successful")
    except Exception as exc:
        log.error("[auto_rename_session] Rename FAILED: %s", exc)


def load_session(session_id: str):
    """Load an existing session from MongoDB."""
    log.info("[load_session] Loading session: %s", session_id)
    result = get_chat_history(session_id)
    if "error" in result:
        log.error("[load_session] Error loading session: %s", result["error"])
        st.error(f"Could not load session: {result['error']}")
        return
    st.session_state.session_id = session_id
    st.session_state.messages = [
        {"role": m["role"], "content": m["content"]}
        for m in result.get("messages", [])
    ]
    st.session_state.openai_history = []
    for m in st.session_state.messages:
        st.session_state.openai_history.append(
            {"role": m["role"], "content": m["content"]}
        )
    log.info("[load_session] Loaded %d messages for session %s", len(st.session_state.messages), session_id)


def persist_message(role: str, content: str):
    """Save a message to both local state and MongoDB."""
    log.info("[persist_message] Saving %s message (%d chars)", role, len(content))
    st.session_state.messages.append({"role": role, "content": content})
    if st.session_state.session_id:
        try:
            save_chat_message(st.session_state.session_id, role, content)
            log.info("[persist_message] Saved to MongoDB")
        except Exception as exc:
            log.error("[persist_message] MongoDB save FAILED: %s", exc)


# ────────────────────────────────────────────
# OpenAI — conversation + tool-use loop
# ────────────────────────────────────────────


def get_ai_response(user_message: str):
    """
    Send the conversation to OpenAI, handle any tool calls (function calling),
    and return the final text reply.
    """
    log.info("[get_ai_response] START — user message: '%s'", user_message[:80])
    client = OpenAI(api_key=st.session_state.openai_key)

    # Add user message to history
    st.session_state.openai_history.append({"role": "user", "content": user_message})

    # Build messages with system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.openai_history
    log.info("[get_ai_response] Sending %d messages to OpenAI (model=gpt-4o-mini)", len(messages))

    # Call OpenAI with tools
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=OPENAI_TOOLS,
    )

    message = response.choices[0].message
    log.info("[get_ai_response] OpenAI responded — has_tool_calls=%s, finish_reason=%s",
             bool(message.tool_calls), response.choices[0].finish_reason)

    # Tool-use loop: OpenAI may request function calls.
    loop_count = 0
    while message.tool_calls:
        loop_count += 1
        log.info("[get_ai_response] Tool-call loop #%d — %d tool calls requested", loop_count, len(message.tool_calls))

        # Add assistant message (with tool_calls) to history
        st.session_state.openai_history.append(message.model_dump())

        # Execute each tool call
        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            log.info("[get_ai_response] Calling tool: %s with args: %s", fn_name, json.dumps(fn_args, default=str)[:200])

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
                log.info("[get_ai_response] Executing %s...", fn_name)
                result = TOOL_FUNCTIONS[fn_name](**fn_args)
                result_data = result if isinstance(result, dict) else {"result": str(result)}
                log.info("[get_ai_response] Tool %s SUCCESS — response keys: %s", fn_name, list(result_data.keys()) if isinstance(result_data, dict) else "n/a")
            except Exception as exc:
                log.error("[get_ai_response] Tool %s FAILED: %s", fn_name, exc)
                result_data = {
                    "error": f"Felix API call failed: {str(exc)}. "
                    "Please inform the user about this error clearly."
                }

            # Add tool result to history
            st.session_state.openai_history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result_data),
            })

        # Get next response
        log.info("[get_ai_response] Sending tool results back to OpenAI...")
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.openai_history
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=OPENAI_TOOLS,
        )
        message = response.choices[0].message
        log.info("[get_ai_response] OpenAI follow-up — has_tool_calls=%s, finish_reason=%s",
                 bool(message.tool_calls), response.choices[0].finish_reason)

    # Add final assistant response to history
    reply = message.content or ""
    st.session_state.openai_history.append({"role": "assistant", "content": reply})

    log.info("[get_ai_response] DONE — reply length: %d chars", len(reply))
    return reply


# ────────────────────────────────────────────
# Custom CSS
# ────────────────────────────────────────────

CUSTOM_CSS = """
<style>
/* Sidebar session buttons */
section[data-testid="stSidebar"] .stButton > button {
    font-size: 0.85rem;
    padding: 0.3rem 0.6rem;
}

/* Chat messages */
.stChatMessage {
    padding: 0.8rem 1rem;
    border-radius: 12px;
    margin-bottom: 0.5rem;
}

/* Make chat input stand out */
.stChatInputContainer {
    border-top: 1px solid rgba(128, 128, 128, 0.2);
    padding-top: 0.5rem;
}
</style>
"""

# ────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────


def main():
    st.set_page_config(
        page_title="Felix Portfolio Advisor",
        page_icon="💬",
        layout="wide",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_session()

    # ── Sidebar ──────────────────────────────
    with st.sidebar:
        st.title("Felix Advisor")

        # Load API key and MongoDB URI from Streamlit Secrets (cloud) or .env (local)
        def _get_secret(key, default=""):
            try:
                val = st.secrets[key]
                log.info("[config] Loaded %s from st.secrets", key)
                return val
            except (KeyError, FileNotFoundError):
                val = os.getenv(key, default)
                if val and val != default:
                    log.info("[config] Loaded %s from .env", key)
                else:
                    log.warning("[config] %s NOT FOUND in secrets or .env, using default", key)
                return val

        st.session_state.openai_key = _get_secret("OPENAI_API_KEY")
        _state["mongo_uri"] = _get_secret("MONGO_URI", "mongodb://localhost:27017")
        log.info("[config] OpenAI key set: %s, MongoDB URI set: %s",
                 bool(st.session_state.openai_key), bool(_state["mongo_uri"]))

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

        if st.button("+ New Chat", use_container_width=True, type="primary"):
            start_new_session()
            st.rerun()

        # Rename dialog
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
            log.info("[sidebar] Fetching chat sessions from MongoDB...")
            sessions = list_chat_sessions(limit=10)
            log.info("[sidebar] Loaded %d sessions", len(sessions))
        except Exception as exc:
            sessions = []
            log.error("[sidebar] MongoDB session list FAILED: %s", exc)
            st.caption("MongoDB not connected — history unavailable")

        for s in sessions:
            sid = s.get("session_id", "")
            label = s.get("user_name", "Guest")
            is_active = sid == st.session_state.session_id
            col1, col2, col3 = st.columns([5, 1, 1])
            with col1:
                btn_label = f"{'> ' if is_active else ''}{label}"
                if st.button(
                    btn_label,
                    key=f"load_{sid}",
                    use_container_width=True,
                    disabled=is_active,
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

        # Auto-rename session from first message
        auto_rename_session(prompt)

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
                    error_msg = str(exc)
                    if "401" in error_msg or "403" in error_msg:
                        st.error("Your Felix Bearer token is invalid or expired. Please update it in the sidebar.")
                    elif "timeout" in error_msg.lower():
                        st.error("The Felix API took too long to respond. Please try again.")
                    elif "api_key" in error_msg.lower() or "auth" in error_msg.lower():
                        st.error("OpenAI API key issue. Please check your key.")
                    else:
                        st.error(f"Something went wrong: {error_msg}")


if __name__ == "__main__":
    main()
