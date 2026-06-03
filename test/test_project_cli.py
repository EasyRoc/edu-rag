"""项目管理 CLI 的轻量回归测试。"""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = PROJECT_ROOT / "scripts" / "edu_rag.py"
ROOT_COMMAND_PATH = PROJECT_ROOT / "edu-rag"


def load_cli_module():
    spec = importlib.util.spec_from_file_location("edu_rag_cli", CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ProjectCliTests(unittest.TestCase):
    def setUp(self):
        self.cli = load_cli_module()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        (self.root / ".env.example").write_text("APP_PORT=8000\n", encoding="utf-8")
        (self.root / "sample_docs").mkdir()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_parser_registers_new_user_and_daily_commands(self):
        parser = self.cli.build_parser()
        help_text = parser.format_help()

        for command in [
            "help",
            "setup",
            "start",
            "stop",
            "restart",
            "status",
            "health",
            "logs",
            "open",
            "list-docs",
            "upload-samples",
            "delete-samples",
            "ask",
            "test",
            "eval-sample",
        ]:
            self.assertIn(command, help_text)

    def test_help_subcommand_prints_operations(self):
        stdout = StringIO()

        with redirect_stdout(stdout):
            exit_code = self.cli.main(["help"])

        output = stdout.getvalue()
        self.assertEqual(0, exit_code)
        self.assertIn("Edu-RAG 本地项目管理 CLI", output)
        self.assertIn("upload-samples", output)
        self.assertIn("delete-samples", output)

    def test_root_command_script_delegates_to_python_cli(self):
        self.assertTrue(ROOT_COMMAND_PATH.exists())
        content = ROOT_COMMAND_PATH.read_text(encoding="utf-8")
        self.assertIn("scripts/edu_rag.py", content)
        self.assertIn('"$@"', content)

    def test_setup_env_creates_env_from_example_without_overwriting_existing_file(self):
        env_path = self.root / ".env"

        created = self.cli.ensure_env_file(self.root)
        self.assertTrue(created)
        self.assertEqual("APP_PORT=8000\n", env_path.read_text(encoding="utf-8"))

        env_path.write_text("APP_PORT=9000\n", encoding="utf-8")
        created_again = self.cli.ensure_env_file(self.root)
        self.assertFalse(created_again)
        self.assertEqual("APP_PORT=9000\n", env_path.read_text(encoding="utf-8"))

    def test_runtime_paths_are_kept_under_run_directory(self):
        paths = self.cli.RuntimePaths.from_root(self.root)

        self.assertEqual(self.root / ".run", paths.run_dir)
        self.assertEqual(self.root / ".run" / "edu-rag.pid", paths.pid_file)
        self.assertEqual(self.root / ".run" / "edu-rag.log", paths.log_file)
        self.assertEqual(self.root / ".run" / "sample-docs.json", paths.sample_manifest)

    def test_sample_manifest_round_trips_uploaded_document_ids(self):
        paths = self.cli.RuntimePaths.from_root(self.root)
        records = [
            {"id": "doc-1", "file": "sample_docs/a.md"},
            {"id": "doc-2", "file": "sample_docs/b.txt"},
        ]

        self.cli.write_sample_manifest(paths, records)

        self.assertEqual(records, self.cli.read_sample_manifest(paths))

    def test_delete_samples_uses_manifest_ids_and_document_endpoint(self):
        paths = self.cli.RuntimePaths.from_root(self.root)
        self.cli.write_sample_manifest(
            paths,
            [
                {"id": "doc-1", "file": "sample_docs/a.md"},
                {"id": "doc-2", "file": "sample_docs/b.txt"},
            ],
        )
        calls = []

        class FakeHttpClient:
            def delete_json(self, url):
                calls.append(url)
                return {"code": 0, "message": "删除成功"}

        deleted = self.cli.delete_sample_documents(
            base_url="http://127.0.0.1:8000",
            paths=paths,
            http_client=FakeHttpClient(),
        )

        self.assertEqual(2, deleted)
        self.assertEqual(
            [
                "http://127.0.0.1:8000/api/v1/documents/doc-1",
                "http://127.0.0.1:8000/api/v1/documents/doc-2",
            ],
            calls,
        )
        self.assertFalse(paths.sample_manifest.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
