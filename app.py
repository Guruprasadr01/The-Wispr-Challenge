import os
from importlib.util import find_spec
from pathlib import Path

import streamlit as st

from debugtrail import (
    DB_PATH,
    create_entry,
    delete_entry,
    init_db,
    list_entries,
    search_entries,
    semantic_search_entries,
    update_entry,
)

_ENTRY_FIELDS = ("id", "title", "error", "root_cause", "solution", "lesson_learned", "tags")


st.set_page_config(
    page_title="DebugTrail — Your debugging memory",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&display=swap');
    :root {
        --ink: #e9edf5;
        --muted: #9ba8bc;
        --panel: #111a29;
        --line: #243248;
        --mint: #71e0bd;
        --blue: #8cb7ff;
    }
    .stApp { background: #0b1220; color: var(--ink); font-family: 'DM Sans', sans-serif; }
    [data-testid="stSidebar"] { background: #0e1726; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--muted); }
    h1, h2, h3 { letter-spacing: -0.035em; }
    h1 { font-size: 2.35rem !important; }
    .eyebrow { color: var(--mint); font: 500 0.72rem 'DM Mono', monospace; letter-spacing: .12em; text-transform: uppercase; }
    .hero { padding: 1.4rem 0 1rem; }
    .hero-copy { color: var(--muted); max-width: 650px; font-size: 1.02rem; line-height: 1.7; }
    .metric-card { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.1rem; }
    .metric-label { color: var(--muted); font-size: .8rem; }
    .metric-value { color: var(--ink); font: 500 1.65rem 'DM Mono', monospace; margin-top: .35rem; }
    .entry-label { color: var(--mint); font: 500 .7rem 'DM Mono', monospace; letter-spacing: .08em; text-transform: uppercase; }
    .stButton > button[kind="primary"] { background: var(--mint); color: #092019; border: 0; font-weight: 700; }
    .stButton > button { border-radius: 9px; }
    [data-testid="stExpander"] { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; }
    div[data-baseweb="input"] input, div[data-baseweb="textarea"] textarea { background: #0c1422; }
    hr { border-color: var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)

init_db(DB_PATH)


@st.cache_resource
def _local_embedding_model():
    from fastembed import TextEmbedding

    model_cache = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "DebugTrail" / "models"
    return TextEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        cache_dir=str(model_cache),
    )


@st.cache_data(max_entries=64, show_spinner="Searching with the local AI model...")
def _cached_semantic_search(
    query: str, snapshots: tuple[tuple[object, ...], ...]
) -> tuple[tuple[int, float], ...]:
    entries = [dict(zip(_ENTRY_FIELDS, snapshot, strict=True)) for snapshot in snapshots]
    matches = semantic_search_entries(query, entries, _local_embedding_model().embed)
    return tuple((int(entry["id"]), score) for entry, score in matches)


def go_to_log_fix() -> None:
    st.session_state["page"] = "Log a fix"


def show_entry(entry: dict[str, object]) -> None:
    st.markdown("**The error**")
    st.write(entry["error"])
    st.markdown("**Root cause**")
    st.write(entry["root_cause"])
    st.markdown("**Solution**")
    st.write(entry["solution"])
    st.markdown("**Lesson learned**")
    st.write(entry["lesson_learned"])
    tags = str(entry["tags"]).strip()
    if tags:
        st.caption(f"Tags · {tags}")
    st.caption(f"Saved {entry['created_at']}")


def entry_editor(entry: dict[str, object]) -> None:
    entry_id = int(entry["id"])
    with st.form(f"edit-entry-{entry_id}"):
        title = st.text_input("Short title", str(entry["title"]))
        error = st.text_area("Error or symptom", str(entry["error"]), height=100)
        root_cause = st.text_area("Root cause", str(entry["root_cause"]), height=100)
        solution = st.text_area("Solution that fixed it", str(entry["solution"]), height=100)
        lesson = st.text_area("Lesson learned", str(entry["lesson_learned"]), height=85)
        tags = st.text_input("Tags (comma separated)", str(entry["tags"]))
        saved = st.form_submit_button("Save changes", type="primary")
    if saved:
        update_entry(entry_id, title, error, root_cause, solution, lesson, tags, DB_PATH)
        st.success("Your debugging note was updated.")
        st.rerun()

    with st.form(f"delete-entry-{entry_id}"):
        confirmed = st.checkbox("I understand this note will be permanently deleted.")
        deleted = st.form_submit_button("Delete note")
    if deleted:
        if not confirmed:
            st.warning("Confirm deletion before removing this note.")
        else:
            delete_entry(entry_id, DB_PATH)
            st.success("Debugging note deleted.")
            st.rerun()


with st.sidebar:
    st.markdown('<div class="eyebrow">DEBUGTRAIL / LOCAL MEMORY</div>', unsafe_allow_html=True)
    st.title("🧭 DebugTrail")
    st.caption("A little less debugging from scratch.")
    page = st.radio(
        "Navigate",
        ["Overview", "Log a fix", "Search memory", "All notes"],
        label_visibility="collapsed",
        key="page",
    )
    st.divider()
    st.caption("Your notes stay in a local SQLite database.")


if page == "Overview":
    entries = list_entries(db_path=DB_PATH)
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">YOUR DEBUGGING MEMORY</div>
          <h1>Every fix makes the next one easier.</h1>
          <div class="hero-copy">Capture what broke, why it broke, and what you learned.
          When a familiar error comes back, search your own trail of fixes.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    for column, label, value in (
        (col1, "Fixes remembered", len(entries)),
        (col2, "Unique tags", len({tag.strip().lower() for item in entries for tag in str(item["tags"]).split(",") if tag.strip()})),
        (col3, "Search mode", "Local & private"),
    ):
        with column:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### Recent trail")
    if entries:
        for entry in entries[:5]:
            with st.expander(f"🧩  {entry['title']}"):
                show_entry(entry)
    else:
        st.info("Your debugging memory is ready. Log your first fix to start building your trail.")
        st.button(
            "＋  Log your first fix",
            type="primary",
            on_click=go_to_log_fix,
        )

elif page == "Log a fix":
    st.markdown('<div class="eyebrow">ADD TO YOUR MEMORY</div>', unsafe_allow_html=True)
    st.title("Log a debugging win")
    st.caption("Write it down while the root cause and fix are still fresh.")

    with st.form("new-entry", clear_on_submit=True):
        title = st.text_input("Short title *", placeholder="e.g. API request failed after token refresh")
        error = st.text_area("Error or symptom *", placeholder="What did you see? Include the message, behavior, and useful context.", height=120)
        root_cause = st.text_area("Root cause *", placeholder="What was actually causing the problem?", height=120)
        solution = st.text_area("Solution that fixed it *", placeholder="What change resolved the issue?", height=120)
        lesson = st.text_area("Lesson learned *", placeholder="What would help you recognize or avoid this next time?", height=100)
        tags = st.text_input("Tags", placeholder="python, api, authentication")
        submitted = st.form_submit_button("Save this fix", type="primary")

    if submitted:
        entry_id = create_entry(title, error, root_cause, solution, lesson, tags, DB_PATH)
        st.success(f"Fix saved to your trail (note #{entry_id}).")
        st.rerun()

elif page == "Search memory":
    st.markdown('<div class="eyebrow">REUSE WHAT YOU LEARNED</div>', unsafe_allow_html=True)
    st.title("Search your debugging memory")
    st.caption("Find past fixes by matching their wording or meaning.")
    use_local_ai = st.toggle(
        "Use local AI semantic search",
        help=(
            "Runs an embedding model on this computer. The model downloads once when "
            "you first search; your notes are not sent to an AI service."
        ),
    )
    query = st.text_input("Search", placeholder="Try: connection timeout, missing key, or token refresh")
    if query.strip():
        if use_local_ai and find_spec("fastembed") is None:
            st.info(
                "Local AI search is optional. Install it with "
                "`python -m pip install -r requirements-ai.txt`, then restart DebugTrail. "
                "Showing keyword matches for now."
            )
            results = search_entries(query, DB_PATH)
        else:
            entries = list_entries(db_path=DB_PATH)
            if use_local_ai and entries:
                snapshots = tuple(
                    tuple(entry[field] for field in _ENTRY_FIELDS) for entry in entries
                )
                scores_by_id = dict(_cached_semantic_search(query, snapshots))
                results = [
                    entry
                    for entry in entries
                    if int(entry["id"]) in scores_by_id
                ]
                results.sort(key=lambda entry: scores_by_id[int(entry["id"])], reverse=True)
                if results:
                    st.caption(f"{len(results)} related note{'s' if len(results) != 1 else ''} · local AI matches")
                    for entry in results:
                        with st.expander(f"🧩  {entry['title']}"):
                            st.caption(
                                f"Semantic similarity · "
                                f"{scores_by_id[int(entry['id'])]:.0%}"
                            )
                            show_entry(entry)
                else:
                    st.info("No notes to compare yet. Log a fix to start your debugging memory.")
                results = None
            else:
                results = search_entries(query, DB_PATH)

        if results is not None:
            if results:
                st.caption(f"{len(results)} related note{'s' if len(results) != 1 else ''} · best matches first")
                for entry in results:
                    with st.expander(f"🧩  {entry['title']}"):
                        show_entry(entry)
            else:
                st.info("No matching notes yet. Try fewer keywords or log this fix for next time.")
    else:
        st.info("Enter a symptom or error phrase to find related fixes.")

else:
    st.markdown('<div class="eyebrow">YOUR PERSONAL KNOWLEDGE BASE</div>', unsafe_allow_html=True)
    st.title("All debugging notes")
    entries = list_entries(db_path=DB_PATH)
    if not entries:
        st.info("No notes yet. Once you log a fix, it will be waiting here.")
    else:
        filter_text = st.text_input("Filter notes", placeholder="Filter by title, error, cause, solution, lesson, or tag")
        visible_entries = search_entries(filter_text, DB_PATH) if filter_text.strip() else entries
        st.caption(f"{len(visible_entries)} of {len(entries)} notes")
        for entry in visible_entries:
            with st.expander(f"🧩  {entry['title']}"):
                show_entry(entry)
                st.divider()
                entry_editor(entry)
