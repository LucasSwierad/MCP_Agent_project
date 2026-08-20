"""
dashboard.py — Streamlit dashboard for the agent's memory.

Run with:
    streamlit run dashboard.py

Reads memory.db (created automatically by memory.py / client.py — see
that module's docstring). This file is read-only: it never writes to
the database, so it's safe to run alongside client.py at the same time.
"""

import os
import json
import asyncio

import pandas as pd
import streamlit as st

from memory import MemoryStore, new_session_id
from mcp import Client
from mcp.client.stdio import stdio_client
from client import get_server_params, process_query


DB_PATH = os.getenv("MEMORY_DB_PATH", "memory.db")

st.set_page_config(page_title="Agent Memory Dashboard", page_icon="🧠", layout="wide")


@st.cache_resource
def get_store() -> MemoryStore:
    return MemoryStore(db_path=DB_PATH)


store = get_store()

if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = new_session_id()
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


async def run_query_once(server_args: list[str], query: str, session_id: str) -> str:
    """Open a fresh MCP connection, run one query through the real pipeline
    (recall -> run_pipeline -> store), close the connection, return the
    response. A new connection per query avoids trying to keep an async
    subprocess connection alive across Streamlit reruns, which happen on
    unpredictable threads (the same class of issue that broke raw sqlite3
    connections here earlier)."""
    params = get_server_params(server_args)
    async with Client(stdio_client(params)) as mcp_client:
        tool_list = await mcp_client.list_tools()
        return await process_query(mcp_client, tool_list.tools, query, session_id)

st.title("🧠 Agent Memory Dashboard")
st.caption(f"Reading from `{DB_PATH}`")

if st.button("🔄 Refresh"):
    st.cache_resource.clear()
    st.rerun()

if not os.path.exists(DB_PATH):
    st.warning(
        f"No database found at `{DB_PATH}` yet. It's created automatically the "
        "first time you run client.py and complete a query."
    )
    st.stop()

# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------

stats = store.get_stats()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Long-term notes", stats["total_notes"])
col2.metric("Sessions", stats["total_sessions"])
col3.metric("Conversation turns", stats["total_turns"])
col4.metric(
    "Semantic-indexed notes",
    stats["embedded_notes"],
    help="Notes with a Gemini embedding stored. If 0, GOOGLE_API_KEY/"
         "GEMINI_API_KEY wasn't set when they were created, so recall for "
         "them falls back to keyword (FTS5) search only.",
)

st.divider()

tab_chat, tab_search, tab_notes, tab_sessions = st.tabs(
    ["🗨️ Chat", "🔍 Search memory", "📚 All long-term notes", "💬 Sessions"]
)

# --------------------------------------------------------------------------
# Chat — actually run queries through the agent (recall -> pipeline -> store)
# --------------------------------------------------------------------------

with tab_chat:
    st.subheader("Chat with the agent")

    with st.sidebar:
        st.subheader("MCP server")
        server_command = st.text_input(
            "Server command",
            value=st.session_state.get("server_command", ""),
            placeholder="npx -y @modelcontextprotocol/server-filesystem ./workspace",
            help="Same command you'd pass to `python client.py ...` — "
                 "space-separated command and args.",
        )
        st.session_state.server_command = server_command
        st.caption(f"Session: `{st.session_state.chat_session_id}`")
        if st.button("New chat session"):
            st.session_state.chat_session_id = new_session_id()
            st.session_state.chat_messages = []
            st.rerun()

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = st.chat_input("Ask the agent...")

    if prompt:
        if not server_command.strip():
            st.error(
                "Set the MCP server command in the sidebar first "
                "(the same one you'd pass to `python client.py ...`)."
            )
        else:
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        server_args = server_command.strip().split()
                        response = asyncio.run(
                            run_query_once(
                                server_args, prompt, st.session_state.chat_session_id
                            )
                        )
                    except Exception as e:
                        response = f"Error: {e}"
                st.write(response)

            st.session_state.chat_messages.append(
                {"role": "assistant", "content": response}
            )
            # New note/turns were just written by process_query — refresh
            # cached stats/tables on the other tabs.
            st.cache_resource.clear()
            st.rerun()

# --------------------------------------------------------------------------
# Search — exercises the same hybrid recall client.py uses
# --------------------------------------------------------------------------

with tab_search:
    st.subheader("Search long-term memory")
    st.caption(
        "Runs the same hybrid recall (semantic + keyword) that client.py "
        "uses before every query."
    )

    query = st.text_input("Query", placeholder="e.g. renewable energy in Germany")
    top_k = st.slider("Results", min_value=1, max_value=10, value=5)

    if query:
        results = store.search_long_term(query, top_k=top_k)
        if not results:
            st.info("No matches.")
        for note in results:
            badge = "🧬 semantic" if note.match_type == "semantic" else "🔤 keyword"
            with st.expander(f"{badge} · score {note.score:.2f} · {note.task}"):
                st.write("**Summary**")
                st.write(note.summary or "_(none)_")
                if note.plan:
                    st.write("**Plan**")
                    st.json(note.plan)
                if note.research_results:
                    st.write("**Research results**")
                    st.json(note.research_results)
                st.caption(f"Note #{note.id} · {note.created_at}")

# --------------------------------------------------------------------------
# All notes — full browsable table
# --------------------------------------------------------------------------

with tab_notes:
    st.subheader("All long-term notes")

    notes = store.get_all_notes(limit=500)
    if not notes:
        st.info("No notes logged yet. Run a query in client.py to create one.")
    else:
        df = pd.DataFrame([
            {
                "ID": n["id"],
                "Created": n["created_at"],
                "Task": n["task"],
                "Summary": (n["summary"][:120] + "…") if len(n["summary"]) > 120 else n["summary"],
                "Status": n["status"],
                "Semantic index": "✅" if n["has_embedding"] else "—",
            }
            for n in notes
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()
        st.write("**Inspect a note**")
        note_ids = [n["id"] for n in notes]
        selected_id = st.selectbox("Note ID", note_ids)
        selected = next(n for n in notes if n["id"] == selected_id)

        st.write("**Task**")
        st.write(selected["task"])
        st.write("**Summary**")
        st.write(selected["summary"] or "_(none)_")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Plan**")
            st.json(selected["plan"])
        with c2:
            st.write("**Research results**")
            st.json(selected["research_results"])
        st.write("**Tools used**")
        st.write(", ".join(selected["tool_calls_log"]) or "_(none)_")

# --------------------------------------------------------------------------
# Sessions — short-term conversation buffers
# --------------------------------------------------------------------------

with tab_sessions:
    st.subheader("Sessions")

    sessions = store.get_sessions()
    if not sessions:
        st.info("No sessions yet.")
    else:
        sdf = pd.DataFrame(sessions).rename(columns={
            "session_id": "Session",
            "turns": "Turns",
            "started_at": "Started",
            "last_active": "Last active",
        })
        st.dataframe(sdf, use_container_width=True, hide_index=True)

        st.divider()
        session_ids = [s["session_id"] for s in sessions]
        selected_session = st.selectbox("View conversation", session_ids)

        turns = store.get_session_turns(selected_session)
        for t in turns:
            role = t["role"]
            with st.chat_message("user" if role == "user" else "assistant"):
                st.write(t["content"])
                st.caption(t["created_at"])