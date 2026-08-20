"""
memory.py — Retrieval-augmented memory for the MCP agent.

Two tiers:

  Short-term memory
      An in-session rolling buffer of (role, content) turns. Cheap, in
      SQLite so it survives a crash mid-session, but scoped to session_id
      and meant to be small (last N turns).

  Long-term memory
      Persistent "research_notes": every completed task, its plan, tool
      results and summary, kept forever (or until pruned). Retrieval is
      hybrid:
        - If GOOGLE_API_KEY is set, notes are embedded with Gemini's
          text-embedding model and retrieved by cosine similarity
          (semantic recall — finds notes about "cut CO2" when you ask
          about "reduce emissions").
        - Always, notes are also indexed in an SQLite FTS5 virtual table
          for fast keyword/BM25 recall. This is the fallback when no API
          key is configured, and it runs alongside embeddings even when
          they are, since keyword hits are useful too (exact names,
          IDs, filenames embeddings can blur).

Everything here is dependency-light on purpose: only the stdlib sqlite3
module is required to run in fallback (keyword-only) mode. Embeddings are
optional and imported lazily so their absence never breaks the module.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


DB_PATH_DEFAULT = "memory.db"


# --------------------------------------------------------------------------
# Embeddings (optional, lazy). Falls back to None if unavailable so the
# rest of the module degrades to keyword-only search instead of crashing.
# --------------------------------------------------------------------------

_embedder = None
_embedder_load_attempted = False


def _get_embedder():
    """Lazily construct a Gemini embeddings client. Returns None if the
    package isn't installed or no API key is configured — callers must
    handle None and fall back to keyword search."""
    global _embedder, _embedder_load_attempted
    if _embedder_load_attempted:
        return _embedder
    _embedder_load_attempted = True

    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        return None

    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        # NOTE: verify this model name against current Gemini API docs —
        # embedding model names/versions do change over time.
        _embedder = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    except Exception:
        _embedder = None

    return _embedder


def _embed(text: str) -> Optional[list[float]]:
    embedder = _get_embedder()
    if embedder is None:
        return None
    try:
        return embedder.embed_query(text)
    except Exception:
        # Any runtime failure (quota, network, bad key) -> degrade
        # gracefully rather than taking down the memory layer.
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class RecalledNote:
    id: int
    task: str
    summary: str
    plan: list
    research_results: list
    created_at: str
    score: float
    match_type: str  # "semantic" | "keyword"


@dataclass
class MemoryStore:
    db_path: str = DB_PATH_DEFAULT
    _conn: sqlite3.Connection = field(init=False, repr=False)

    def __post_init__(self):
        # check_same_thread=False: needed because Streamlit (and some other
        # callers) may cache this object as a resource and invoke it from a
        # different thread than the one that created the connection across
        # reruns. Safe here since usage is either single-threaded (client.py)
        # or read-only (dashboard.py) — no concurrent writers from multiple
        # threads.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # ---- schema -----------------------------------------------------

    def _init_schema(self) -> None:
        cur = self._conn.cursor()

        # Long-term memory: one row per completed task.
        cur.execute("""
        CREATE TABLE IF NOT EXISTS research_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            status TEXT NOT NULL,
            task TEXT NOT NULL,
            plan TEXT NOT NULL,
            research_results TEXT,
            summary TEXT,
            tool_calls_log TEXT,
            embedding TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Keyword index over the same notes (FTS5). `content` +
        # `content_rowid` mirrors research_notes.id so we can join back.
        cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS research_notes_fts
        USING fts5(task, summary, content='research_notes', content_rowid='id');
        """)

        # Keep the FTS index in sync automatically.
        cur.execute("""
        CREATE TRIGGER IF NOT EXISTS research_notes_ai
        AFTER INSERT ON research_notes BEGIN
            INSERT INTO research_notes_fts(rowid, task, summary)
            VALUES (new.id, new.task, new.summary);
        END;
        """)

        # Short-term memory: rolling per-session conversation buffer.
        cur.execute("""
        CREATE TABLE IF NOT EXISTS session_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """)
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_session_history_session
        ON session_history(session_id, id);
        """)

        self._conn.commit()

    # ---- long-term: write --------------------------------------------

    def add_long_term(
        self,
        task: str,
        plan: list,
        research_results: list,
        summary: str,
        tool_calls_log: list,
        status: str = "COMPLETED",
    ) -> int:
        """Persist a completed task and index it for both semantic and
        keyword recall. Returns the new row id."""
        embedding = _embed(f"{task}\n{summary}")

        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO research_notes (
                status, task, plan, research_results, summary,
                tool_calls_log, embedding
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                status,
                task,
                json.dumps(plan or []),
                json.dumps(research_results or []),
                summary or "",
                json.dumps(tool_calls_log or []),
                json.dumps(embedding) if embedding is not None else None,
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    # ---- long-term: read / retrieval ----------------------------------

    def search_long_term(self, query: str, top_k: int = 3) -> list[RecalledNote]:
        """Hybrid recall: semantic (if embeddings configured) merged with
        keyword/FTS5 results, deduplicated, best score wins, capped at
        top_k."""
        results: dict[int, RecalledNote] = {}

        semantic = self._search_semantic(query, top_k)
        for note in semantic:
            results[note.id] = note

        keyword = self._search_keyword(query, top_k)
        for note in keyword:
            if note.id not in results or note.score > results[note.id].score:
                # Don't let a keyword hit overwrite a stronger semantic
                # hit's match_type label, just merge in whichever is better.
                if note.id in results and results[note.id].match_type == "semantic":
                    continue
                results[note.id] = note

        ranked = sorted(results.values(), key=lambda n: n.score, reverse=True)
        return ranked[:top_k]

    def _search_semantic(self, query: str, top_k: int) -> list[RecalledNote]:
        query_vec = _embed(query)
        if query_vec is None:
            return []

        cur = self._conn.cursor()
        cur.execute(
            "SELECT id, task, summary, plan, research_results, created_at, embedding "
            "FROM research_notes WHERE embedding IS NOT NULL"
        )
        scored = []
        for row in cur.fetchall():
            try:
                vec = json.loads(row["embedding"])
            except (TypeError, json.JSONDecodeError):
                continue
            score = _cosine(query_vec, vec)
            scored.append(
                RecalledNote(
                    id=row["id"],
                    task=row["task"],
                    summary=row["summary"] or "",
                    plan=json.loads(row["plan"] or "[]"),
                    research_results=json.loads(row["research_results"] or "[]"),
                    created_at=row["created_at"],
                    score=score,
                    match_type="semantic",
                )
            )
        scored.sort(key=lambda n: n.score, reverse=True)
        return scored[:top_k]

    def _search_keyword(self, query: str, top_k: int) -> list[RecalledNote]:
        # FTS5 MATCH needs a query string; sanitize naively by quoting
        # each token so punctuation in the user's query can't break syntax.
        tokens = [t for t in query.replace('"', " ").split() if t]
        if not tokens:
            return []
        fts_query = " OR ".join(f'"{t}"' for t in tokens)

        cur = self._conn.cursor()
        try:
            cur.execute(
                """
                SELECT rn.id, rn.task, rn.summary, rn.plan, rn.research_results,
                       rn.created_at, bm25(research_notes_fts) AS rank
                FROM research_notes_fts
                JOIN research_notes rn ON rn.id = research_notes_fts.rowid
                WHERE research_notes_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, top_k),
            )
        except sqlite3.OperationalError:
            return []

        notes = []
        for row in cur.fetchall():
            # bm25() is negative-is-better; flip and normalize loosely to
            # a 0..1-ish range so it can compare against cosine scores.
            raw_rank = row["rank"]
            score = 1.0 / (1.0 + max(0.0, -raw_rank))
            notes.append(
                RecalledNote(
                    id=row["id"],
                    task=row["task"],
                    summary=row["summary"] or "",
                    plan=json.loads(row["plan"] or "[]"),
                    research_results=json.loads(row["research_results"] or "[]"),
                    created_at=row["created_at"],
                    score=score,
                    match_type="keyword",
                )
            )
        return notes

    # ---- short-term: write / read -------------------------------------

    def add_short_term(self, session_id: str, role: str, content: str) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO session_history (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        self._conn.commit()

    def get_short_term(self, session_id: str, limit: int = 10) -> list[dict]:
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT role, content FROM session_history
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        )
        rows = cur.fetchall()
        rows.reverse()  # chronological order
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    # ---- context assembly ----------------------------------------------

    def build_context(self, session_id: str, query: str, top_k: int = 3, turns: int = 6) -> str:
        """Assemble a single context string combining recent conversation
        and relevant long-term recall, ready to hand to the pipeline."""
        parts = []

        short_term = self.get_short_term(session_id, limit=turns)
        if short_term:
            convo = "\n".join(f"{t['role']}: {t['content']}" for t in short_term)
            parts.append(f"## Recent conversation\n{convo}")

        recalled = self.search_long_term(query, top_k=top_k)
        if recalled:
            notes = []
            for n in recalled:
                notes.append(
                    f"- [{n.match_type}, score={n.score:.2f}] Task: {n.task}\n"
                    f"  Summary: {n.summary}"
                )
            parts.append("## Relevant past tasks\n" + "\n".join(notes))

        return "\n\n".join(parts) if parts else ""

    # ---- read-only helpers for dashboards / inspection -----------------

    def get_stats(self) -> dict:
        """Summary counts for a dashboard header."""
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM research_notes")
        total_notes = cur.fetchone()[0]

        cur.execute("SELECT COUNT(DISTINCT session_id) FROM session_history")
        total_sessions = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM session_history")
        total_turns = cur.fetchone()[0]

        cur.execute(
            "SELECT COUNT(*) FROM research_notes WHERE embedding IS NOT NULL"
        )
        embedded_notes = cur.fetchone()[0]

        return {
            "total_notes": total_notes,
            "total_sessions": total_sessions,
            "total_turns": total_turns,
            "embedded_notes": embedded_notes,
        }

    def get_all_notes(self, limit: int = 200) -> list[dict]:
        """All long-term notes, most recent first, as plain dicts (safe to
        hand straight to pandas/streamlit)."""
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT id, status, task, plan, research_results, summary,
                   tool_calls_log, embedding IS NOT NULL AS has_embedding,
                   created_at
            FROM research_notes
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        notes = []
        for row in cur.fetchall():
            notes.append({
                "id": row["id"],
                "status": row["status"],
                "task": row["task"],
                "plan": json.loads(row["plan"] or "[]"),
                "research_results": json.loads(row["research_results"] or "[]"),
                "summary": row["summary"] or "",
                "tool_calls_log": json.loads(row["tool_calls_log"] or "[]"),
                "has_embedding": bool(row["has_embedding"]),
                "created_at": row["created_at"],
            })
        return notes

    def get_sessions(self) -> list[dict]:
        """One row per session: id, turn count, first/last activity."""
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT session_id,
                   COUNT(*) AS turns,
                   MIN(created_at) AS started_at,
                   MAX(created_at) AS last_active
            FROM session_history
            GROUP BY session_id
            ORDER BY last_active DESC
            """
        )
        return [dict(row) for row in cur.fetchall()]

    def get_session_turns(self, session_id: str) -> list[dict]:
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT role, content, created_at FROM session_history
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        )
        return [dict(row) for row in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()


def new_session_id() -> str:
    return f"{int(time.time())}-{uuid.uuid4().hex[:8]}"