import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence


DEFAULT_DB_PATH = Path(__file__).with_name("debugtrail.db")
DB_PATH = Path(os.environ.get("DEBUGTRAIL_DB_PATH", DEFAULT_DB_PATH))

_FIELDS = ("title", "error", "root_cause", "solution", "lesson_learned", "tags")


@contextmanager
def _connect(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db(db_path: str | Path = DB_PATH) -> None:
    path = Path(db_path)
    if str(db_path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)

    with _connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS debug_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                error TEXT NOT NULL,
                root_cause TEXT NOT NULL,
                solution TEXT NOT NULL,
                lesson_learned TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_debug_entries_created_at "
            "ON debug_entries(created_at DESC)"
        )


def create_entry(
    title: str,
    error: str,
    root_cause: str,
    solution: str,
    lesson_learned: str,
    tags: str = "",
    db_path: str | Path = DB_PATH,
) -> int:
    values = (title, error, root_cause, solution, lesson_learned)
    if any(not value.strip() for value in values):
        raise ValueError("Title, error, root cause, solution, and lesson learned are required.")

    with _connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO debug_entries
                (title, error, root_cause, solution, lesson_learned, tags)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (*[value.strip() for value in values], tags.strip()),
        )
        return int(cursor.lastrowid)


def update_entry(
    entry_id: int,
    title: str,
    error: str,
    root_cause: str,
    solution: str,
    lesson_learned: str,
    tags: str = "",
    db_path: str | Path = DB_PATH,
) -> None:
    values = (title, error, root_cause, solution, lesson_learned)
    if any(not value.strip() for value in values):
        raise ValueError("Title, error, root cause, solution, and lesson learned are required.")

    with _connect(db_path) as connection:
        cursor = connection.execute(
            """
            UPDATE debug_entries
            SET title = ?, error = ?, root_cause = ?, solution = ?,
                lesson_learned = ?, tags = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (*[value.strip() for value in values], tags.strip(), entry_id),
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Debug entry {entry_id} was not found.")


def delete_entry(entry_id: int, db_path: str | Path = DB_PATH) -> None:
    with _connect(db_path) as connection:
        cursor = connection.execute("DELETE FROM debug_entries WHERE id = ?", (entry_id,))
        if cursor.rowcount == 0:
            raise ValueError(f"Debug entry {entry_id} was not found.")


def get_entry(entry_id: int, db_path: str | Path = DB_PATH) -> dict[str, object] | None:
    with _connect(db_path) as connection:
        row = connection.execute(
            "SELECT * FROM debug_entries WHERE id = ?", (entry_id,)
        ).fetchone()
    return dict(row) if row is not None else None


def list_entries(
    limit: int | None = None, db_path: str | Path = DB_PATH
) -> list[dict[str, object]]:
    query = "SELECT * FROM debug_entries ORDER BY created_at DESC, id DESC"
    params: tuple[int, ...] = ()
    if limit is not None:
        if limit < 0:
            raise ValueError("Limit cannot be negative.")
        query += " LIMIT ?"
        params = (limit,)

    with _connect(db_path) as connection:
        rows = connection.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def search_entries(query: str, db_path: str | Path = DB_PATH) -> list[dict[str, object]]:
    terms = list(dict.fromkeys(re.findall(r"[\w%_\\-]+", query.casefold())))
    if not terms:
        return []

    conditions: list[str] = []
    params: list[str] = []
    for term in terms:
        escaped_term = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped_term}%"
        conditions.append(
            "(" + " OR ".join(f"{field} LIKE ? ESCAPE '\\'" for field in _FIELDS) + ")"
        )
        params.extend([pattern] * len(_FIELDS))

    with _connect(db_path) as connection:
        rows = connection.execute(
            f"SELECT * FROM debug_entries WHERE {' OR '.join(conditions)}",
            params,
        ).fetchall()

    results = [dict(row) for row in rows]

    def relevance(entry: dict[str, object]) -> tuple[int, str, int]:
        score = 0
        weights = {
            "title": 5,
            "error": 4,
            "root_cause": 3,
            "solution": 2,
            "lesson_learned": 2,
            "tags": 1,
        }
        for term in terms:
            for field, weight in weights.items():
                if term in str(entry[field]).casefold():
                    score += weight
        return (-score, str(entry["created_at"]), -int(entry["id"]))

    results.sort(key=relevance)
    return results


def entry_search_text(entry: dict[str, object]) -> str:
    return "\n".join(f"{field}: {entry[field]}" for field in _FIELDS)


def semantic_search_entries(
    query: str,
    entries: list[dict[str, object]],
    embed: Callable[[list[str]], Iterable[Sequence[float]]],
) -> list[tuple[dict[str, object], float]]:
    if not query.strip() or not entries:
        return []

    query_prefix = "Represent this sentence for searching relevant passages: "
    vectors = list(
        embed([query_prefix + query.strip(), *(entry_search_text(entry) for entry in entries)])
    )
    if len(vectors) != len(entries) + 1:
        raise ValueError("The embedding model returned an unexpected number of vectors.")

    query_vector = vectors[0]
    query_norm = sum(float(value) ** 2 for value in query_vector) ** 0.5

    matches: list[tuple[dict[str, object], float]] = []
    for entry, vector in zip(entries, vectors[1:], strict=True):
        if len(query_vector) != len(vector):
            raise ValueError("The embedding model returned vectors with inconsistent dimensions.")
        vector_norm = sum(float(value) ** 2 for value in vector) ** 0.5
        if query_norm == 0 or vector_norm == 0:
            similarity = 0.0
        else:
            similarity = sum(
                float(query_value) * float(entry_value)
                for query_value, entry_value in zip(query_vector, vector, strict=True)
            ) / (query_norm * vector_norm)
        matches.append((entry, similarity))

    matches.sort(key=lambda match: (-match[1], -int(match[0]["id"])))
    return matches
