# DebugTrail

DebugTrail is a local-first debugging memory assistant. Record the error, root
cause, solution, and lesson learned for each fix, then search those notes when a
similar problem appears again.

## Run it

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The SQLite database is created next to `app.py` as `debugtrail.db`. To store it
elsewhere, set `DEBUGTRAIL_DB_PATH` before starting Streamlit:

```powershell
$env:DEBUGTRAIL_DB_PATH = "C:\Users\you\Documents\debugtrail.db"
python -m streamlit run app.py
```

## Features

- Capture an error, root cause, solution, lesson learned, and optional tags.
- Search and rank matching notes across all of their content.
- Browse, edit, and delete saved notes.
- Keep data in a local SQLite database; no account or AI API key is needed.
- Optionally enable local AI semantic search with a downloaded embedding model.

Search is deliberately useful without a network connection: it finds keyword
matches and ranks notes by how strongly the title, error, and cause match.

## Optional local AI search

DebugTrail can also compare the meaning of a search with your saved fixes using
a small local embedding model. This does not use a hosted AI API, require an API
key, or send your notes to an AI service. Keyword search remains available
without the optional package.

Install the optional dependency from the project folder:

```powershell
python -m pip install -r requirements-ai.txt
```

Restart DebugTrail with `python -m streamlit run app.py`, open **Search memory**,
and enable **Use local AI semantic search**. The first semantic search downloads
the model to your computer and therefore needs an internet connection. After
the download, the model runs locally.
