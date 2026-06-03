# Project CLI Manager Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a friendly command-line manager so new users can initialize, start, stop, inspect, and demo the Edu-RAG service without remembering raw API calls.

**Architecture:** Create a standard-library Python CLI at `scripts/edu_rag.py`. The script manages local process state in `.run/`, delegates service startup to `python main.py`, and uses HTTP calls for document and QA operations.

**Tech Stack:** Python `argparse`, `subprocess`, `urllib`, `json`, `unittest`, existing FastAPI routes.

---

### Task 1: CLI Behavior Tests

**Files:**
- Create: `test/test_project_cli.py`

- [ ] Add tests for command registration, runtime path resolution, env-file setup, sample manifest write/read, and document deletion URL construction.
- [ ] Run `python test/test_project_cli.py` and confirm it fails because `scripts/edu_rag.py` does not exist yet.

### Task 2: CLI Implementation

**Files:**
- Create: `scripts/edu_rag.py`

- [ ] Implement subcommands: `setup`, `start`, `stop`, `restart`, `status`, `health`, `logs`, `open`, `list-docs`, `upload-samples`, `delete-samples`, `ask`, `test`, and `eval-sample`.
- [ ] Keep local runtime files under `.run/`: PID, log, and sample upload manifest.
- [ ] Avoid third-party CLI dependencies; use only Python standard library.

### Task 3: Docs And Ignore Rules

**Files:**
- Modify: `README.md`
- Modify: `.gitignore`

- [ ] Document the new first-run flow with `python scripts/edu_rag.py setup`, `start`, `upload-samples`, and `ask`.
- [ ] Ignore `.run/` so local process state is never committed.

### Task 4: Verification

**Files:**
- Test: `test/test_project_cli.py`

- [ ] Run `python test/test_project_cli.py`.
- [ ] Run existing focused suites that cover startup-adjacent behavior.
- [ ] Run syntax checking for edited project directories.
