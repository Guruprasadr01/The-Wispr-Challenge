import sqlite3

import pytest

from debugtrail import (
    create_entry,
    delete_entry,
    get_entry,
    init_db,
    list_entries,
    search_entries,
    semantic_search_entries,
    update_entry,
)


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "nested" / "debugtrail.db"
    init_db(path)
    return path


def test_create_and_get_entry(database):
    entry_id = create_entry(
        "Token refresh failed",
        "HTTP 401 after refresh",
        "Refresh token was not persisted",
        "Save the rotated token before retrying",
        "Token rotation can invalidate the previous token",
        "api, auth",
        database,
    )

    entry = get_entry(entry_id, database)
    assert entry is not None
    assert entry["title"] == "Token refresh failed"
    assert entry["tags"] == "api, auth"
    assert len(list_entries(db_path=database)) == 1


def test_search_ranks_title_and_error_matches(database):
    create_entry(
        "Unrelated database issue",
        "Database connection refused",
        "The local server was stopped",
        "Restart the local server",
        "Check service health first",
        db_path=database,
    )
    best_match_id = create_entry(
        "API token refresh returns 401",
        "HTTP 401 after token refresh",
        "The rotated token was not persisted",
        "Persist the new token before retrying",
        "Refresh tokens may invalidate the previous token",
        "api, auth",
        database,
    )

    matches = search_entries("token refresh", database)
    assert [entry["id"] for entry in matches] == [best_match_id]


def test_search_escapes_like_wildcards(database):
    exact_id = create_entry(
        "Percent sign in message",
        "Error rate is 100% after deploy",
        "Bad threshold",
        "Correct threshold",
        "Validate percentages",
        db_path=database,
    )
    create_entry(
        "Other deploy issue",
        "Error rate is 1000 after deploy",
        "Bad threshold",
        "Correct threshold",
        "Validate values",
        db_path=database,
    )

    matches = search_entries("100%", database)
    assert [entry["id"] for entry in matches] == [exact_id]


def test_update_and_delete_entry(database):
    entry_id = create_entry("Old title", "error", "cause", "fix", "lesson", db_path=database)
    update_entry(entry_id, "New title", "error", "cause", "fix", "lesson", db_path=database)
    assert get_entry(entry_id, database)["title"] == "New title"

    delete_entry(entry_id, database)
    assert get_entry(entry_id, database) is None
    with pytest.raises(ValueError, match="not found"):
        delete_entry(entry_id, database)


def test_required_fields_are_enforced(database):
    with pytest.raises(ValueError, match="required"):
        create_entry("Title", "", "Cause", "Fix", "Lesson", db_path=database)
    assert list_entries(db_path=database) == []


def test_database_is_sqlite(database):
    with sqlite3.connect(database) as connection:
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'debug_entries'"
        ).fetchone()
    assert table == ("debug_entries",)


def test_semantic_search_ranks_by_embedding_similarity():
    entries = [
        {
            "id": 1,
            "title": "Database connection exhausted",
            "error": "Could not acquire a connection from the pool",
            "root_cause": "The pool limit was too low",
            "solution": "Increase the connection pool size",
            "lesson_learned": "Monitor concurrent database requests",
            "tags": "database",
        },
        {
            "id": 2,
            "title": "Invalid date format",
            "error": "Could not parse the submitted date",
            "root_cause": "The client sent a localized date",
            "solution": "Normalize dates to ISO 8601",
            "lesson_learned": "Validate date formats at the API boundary",
            "tags": "dates",
        },
    ]

    def fake_embed(texts):
        return [
            [1.0, 0.0]
            if "database connections" in text.casefold()
            or "database connection exhausted" in text.casefold()
            else [0.0, 1.0]
            for text in texts
        ]

    matches = semantic_search_entries("database connections", entries, fake_embed)

    assert [entry["id"] for entry, _ in matches] == [1, 2]
    assert matches[0][1] == pytest.approx(1.0)


def test_semantic_search_skips_embedding_for_empty_input():
    def unexpected_embed(_texts):
        raise AssertionError("Embedding should not run for empty input.")

    assert semantic_search_entries("   ", [], unexpected_embed) == []


def test_semantic_search_rejects_inconsistent_vector_dimensions():
    entry = {
        "id": 1,
        "title": "Title",
        "error": "Error",
        "root_cause": "Cause",
        "solution": "Solution",
        "lesson_learned": "Lesson",
        "tags": "",
    }

    with pytest.raises(ValueError, match="inconsistent dimensions"):
        semantic_search_entries(
            "query",
            [entry],
            lambda _texts: [[1.0, 0.0], [1.0]],
        )
