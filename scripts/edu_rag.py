#!/usr/bin/env python3
"""Edu-RAG 本地项目管理脚本。

这个脚本面向第一次 clone 项目的用户：把环境初始化、服务启停、样例文档
上传/删除、健康检查和常用验证命令收敛成一个统一入口。
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
import webbrowser
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMAND_NAME = "./edu-rag"
DEFAULT_TIMEOUT = 30
SUPPORTED_SAMPLE_SUFFIXES = {".pdf", ".md", ".txt"}


class RuntimePaths:
    """集中管理 CLI 产生的本地运行状态文件。"""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.run_dir = self.root / ".run"
        self.pid_file = self.run_dir / "edu-rag.pid"
        self.log_file = self.run_dir / "edu-rag.log"
        self.sample_manifest = self.run_dir / "sample-docs.json"

    @classmethod
    def from_root(cls, root: Path) -> "RuntimePaths":
        return cls(root)

    def ensure(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)


class HttpClient:
    """基于标准库的轻量 HTTP 客户端，避免给 CLI 增加额外依赖。"""

    def __init__(self, timeout: int = DEFAULT_TIMEOUT):
        self.timeout = timeout

    def get_json(self, url: str) -> dict[str, Any]:
        return self._request_json(url)

    def post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return self._request_json(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )

    def delete_json(self, url: str) -> dict[str, Any]:
        return self._request_json(url, method="DELETE")

    def upload_file(
        self,
        url: str,
        file_path: Path,
        *,
        subject: str,
        grade: str,
        chapter: str,
        strategy: str,
    ) -> dict[str, Any]:
        boundary = f"----edu-rag-{uuid.uuid4().hex}"
        fields = {
            "subject": subject,
            "grade": grade,
            "chapter": chapter,
            "strategy": strategy,
        }
        body = self._build_multipart_body(boundary, fields, file_path)
        return self._request_json(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )

    def _request_json(
        self,
        url: str,
        *,
        data: bytes | None = None,
        method: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        request = urllib.request.Request(url, data=data, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {raw}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"无法连接服务: {exc.reason}") from exc

        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"接口返回不是 JSON: {raw[:200]}") from exc

    @staticmethod
    def _build_multipart_body(boundary: str, fields: dict[str, str], file_path: Path) -> bytes:
        lines: list[bytes] = []
        for name, value in fields.items():
            lines.append(f"--{boundary}\r\n".encode("utf-8"))
            lines.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
            lines.append(str(value).encode("utf-8"))
            lines.append(b"\r\n")

        mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        lines.append(f"--{boundary}\r\n".encode("utf-8"))
        lines.append(
            (
                f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
                f"Content-Type: {mime_type}\r\n\r\n"
            ).encode("utf-8")
        )
        lines.append(file_path.read_bytes())
        lines.append(b"\r\n")
        lines.append(f"--{boundary}--\r\n".encode("utf-8"))
        return b"".join(lines)


def read_env_file(root: Path) -> dict[str, str]:
    env_path = root / ".env"
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def build_base_url(root: Path, explicit_url: str | None = None) -> str:
    if explicit_url:
        return explicit_url.rstrip("/")

    file_env = read_env_file(root)
    port = os.getenv("APP_PORT") or file_env.get("APP_PORT") or "8000"
    host = os.getenv("APP_CLIENT_HOST") or file_env.get("APP_CLIENT_HOST") or "127.0.0.1"
    return f"http://{host}:{port}".rstrip("/")


def ensure_env_file(root: Path) -> bool:
    env_path = root / ".env"
    example_path = root / ".env.example"
    if env_path.exists():
        return False
    if not example_path.exists():
        raise FileNotFoundError("缺少 .env.example，无法生成 .env")
    shutil.copyfile(example_path, env_path)
    return True


def venv_python(root: Path) -> Path:
    if os.name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def resolve_python(root: Path) -> str:
    candidate = venv_python(root)
    if candidate.exists():
        return str(candidate)
    return sys.executable


def ensure_virtualenv(root: Path) -> bool:
    python_path = venv_python(root)
    if python_path.exists():
        return False
    subprocess.run([sys.executable, "-m", "venv", str(root / ".venv")], cwd=root, check=True)
    return True


def install_requirements(root: Path) -> None:
    requirements = root / "requirements.txt"
    if not requirements.exists():
        raise FileNotFoundError("缺少 requirements.txt")
    subprocess.run([resolve_python(root), "-m", "pip", "install", "-r", str(requirements)], cwd=root, check=True)


def read_pid(paths: RuntimePaths) -> int | None:
    if not paths.pid_file.exists():
        return None
    try:
        return int(paths.pid_file.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def process_is_running(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def remove_stale_pid(paths: RuntimePaths) -> None:
    if paths.pid_file.exists():
        paths.pid_file.unlink()


def tail_lines(path: Path, line_count: int) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def wait_for_health(base_url: str, pid: int, timeout: int) -> bool:
    client = HttpClient(timeout=2)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not process_is_running(pid):
            return False
        try:
            result = client.get_json(f"{base_url}/health")
            if result.get("status") == "healthy":
                return True
        except RuntimeError:
            pass
        time.sleep(1)
    return False


def read_sample_manifest(paths: RuntimePaths) -> list[dict[str, Any]]:
    if not paths.sample_manifest.exists():
        return []
    data = json.loads(paths.sample_manifest.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def write_sample_manifest(paths: RuntimePaths, records: list[dict[str, Any]]) -> None:
    paths.ensure()
    paths.sample_manifest.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def delete_sample_documents(base_url: str, paths: RuntimePaths, http_client: HttpClient) -> int:
    records = read_sample_manifest(paths)
    if not records:
        return 0

    deleted = 0
    remaining: list[dict[str, Any]] = []
    for record in records:
        doc_id = record.get("id")
        if not doc_id:
            continue
        try:
            http_client.delete_json(f"{base_url}/api/v1/documents/{doc_id}")
            deleted += 1
        except RuntimeError as exc:
            print(f"删除失败: {doc_id} ({exc})")
            remaining.append(record)

    if remaining:
        write_sample_manifest(paths, remaining)
    elif paths.sample_manifest.exists():
        paths.sample_manifest.unlink()
    return deleted


def infer_sample_metadata(file_path: Path, subject_override: str | None, grade: str) -> dict[str, str]:
    name = file_path.name
    if subject_override:
        subject = subject_override
    elif "数学" in name:
        subject = "数学"
    elif "语文" in name:
        subject = "语文"
    elif "物理" in name:
        subject = "物理"
    else:
        subject = "通用"

    return {
        "subject": subject,
        "grade": grade,
        "chapter": file_path.stem,
    }


def find_sample_files(root: Path) -> list[Path]:
    sample_dir = root / "sample_docs"
    if not sample_dir.exists():
        return []
    return sorted(
        path
        for path in sample_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SAMPLE_SUFFIXES
    )


def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def print_document_list(response: dict[str, Any]) -> None:
    docs = response.get("data") or []
    print(f"文档数量: {response.get('total', len(docs))}")
    if not docs:
        return
    for doc in docs:
        print(
            f"- {doc.get('id', '')} | {doc.get('title', '')} | "
            f"{doc.get('subject', '')} {doc.get('grade', '')} | "
            f"chunks={doc.get('chunk_count', 0)} | {doc.get('status', '')}"
        )


def cmd_setup(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    paths = RuntimePaths.from_root(root)
    paths.ensure()

    env_created = ensure_env_file(root)
    print("已生成 .env" if env_created else ".env 已存在，跳过覆盖")

    if not args.skip_venv:
        created = ensure_virtualenv(root)
        print("已创建 .venv" if created else ".venv 已存在，跳过创建")

    if not args.skip_install:
        print("正在安装依赖，这一步首次运行可能较慢...")
        install_requirements(root)
        print("依赖安装完成")

    print("\n下一步:")
    print(f"  {COMMAND_NAME} start")
    print(f"  {COMMAND_NAME} upload-samples")
    print(f"  {COMMAND_NAME} ask \"一元一次方程怎么解？\" --subject 数学 --grade 七年级")
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    paths = RuntimePaths.from_root(root)
    paths.ensure()
    pid = read_pid(paths)
    if process_is_running(pid):
        print(f"服务已在运行: pid={pid}")
        return 0
    remove_stale_pid(paths)

    log_file = paths.log_file.open("a", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        [resolve_python(root), "main.py"],
        cwd=root,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=(os.name != "nt"),
    )
    log_file.close()
    paths.pid_file.write_text(str(process.pid), encoding="utf-8")
    print(f"服务启动中: pid={process.pid}")
    print(f"日志文件: {paths.log_file}")

    if args.wait:
        base_url = build_base_url(root, args.url)
        if wait_for_health(base_url, process.pid, args.timeout):
            print(f"服务已就绪: {base_url}")
            return 0
        if process.poll() is not None:
            print("服务启动失败，最近日志:")
            print(tail_lines(paths.log_file, 80))
            return process.returncode or 1
        print(f"服务仍在初始化，可稍后运行: {COMMAND_NAME} health")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    paths = RuntimePaths.from_root(root)
    pid = read_pid(paths)
    if not process_is_running(pid):
        remove_stale_pid(paths)
        print("服务未运行")
        return 0

    assert pid is not None
    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        if not process_is_running(pid):
            remove_stale_pid(paths)
            print("服务已停止")
            return 0
        time.sleep(0.2)

    if args.force:
        os.kill(pid, signal.SIGKILL)
        remove_stale_pid(paths)
        print("服务已强制停止")
        return 0

    print(f"服务未在 {args.timeout} 秒内停止，可追加 --force")
    return 1


def cmd_restart(args: argparse.Namespace) -> int:
    stop_args = argparse.Namespace(root=args.root, timeout=args.timeout, force=args.force)
    stop_code = cmd_stop(stop_args)
    if stop_code != 0:
        return stop_code
    start_args = argparse.Namespace(root=args.root, wait=args.wait, timeout=args.timeout, url=args.url)
    return cmd_start(start_args)


def cmd_status(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    paths = RuntimePaths.from_root(root)
    pid = read_pid(paths)
    if process_is_running(pid):
        print(f"运行中: pid={pid}")
    else:
        print("未运行")
        remove_stale_pid(paths)
    print(f"日志文件: {paths.log_file}")
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    base_url = build_base_url(Path(args.root).resolve(), args.url)
    print_json(HttpClient(timeout=args.timeout).get_json(f"{base_url}/health"))
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    paths = RuntimePaths.from_root(Path(args.root).resolve())
    if not paths.log_file.exists():
        print(f"日志文件不存在: {paths.log_file}")
        return 0

    print(tail_lines(paths.log_file, args.lines))
    if args.follow:
        with paths.log_file.open("r", encoding="utf-8", errors="replace") as file:
            file.seek(0, os.SEEK_END)
            try:
                while True:
                    line = file.readline()
                    if line:
                        print(line, end="")
                    else:
                        time.sleep(0.5)
            except KeyboardInterrupt:
                return 0
    return 0


def cmd_open(args: argparse.Namespace) -> int:
    base_url = build_base_url(Path(args.root).resolve(), args.url)
    target = f"{base_url}/docs" if args.docs else base_url
    webbrowser.open(target)
    print(f"已打开: {target}")
    return 0


def cmd_list_docs(args: argparse.Namespace) -> int:
    base_url = build_base_url(Path(args.root).resolve(), args.url)
    response = HttpClient(timeout=args.timeout).get_json(f"{base_url}/api/v1/documents/list")
    if args.json:
        print_json(response)
    else:
        print_document_list(response)
    return 0


def cmd_upload_samples(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    paths = RuntimePaths.from_root(root)
    files = find_sample_files(root)
    if not files:
        print("未找到 sample_docs 下的样例文档")
        return 1

    base_url = build_base_url(root, args.url)
    client = HttpClient(timeout=args.timeout)
    records = read_sample_manifest(paths)
    existing_files = {record.get("file") for record in records}

    for file_path in files:
        relative_file = str(file_path.relative_to(root))
        if relative_file in existing_files and not args.force:
            print(f"跳过已记录样例: {relative_file}")
            continue

        metadata = infer_sample_metadata(file_path, args.subject, args.grade)
        print(f"上传样例: {relative_file} ({metadata['subject']} {metadata['grade']})")
        response = client.upload_file(
            f"{base_url}/api/v1/documents/upload",
            file_path,
            subject=metadata["subject"],
            grade=metadata["grade"],
            chapter=metadata["chapter"],
            strategy=args.strategy,
        )
        if response.get("code", 0) != 0:
            print_json(response)
            continue

        data = response.get("data") or {}
        doc_id = data.get("doc_id") or data.get("id")
        if not doc_id:
            print("接口未返回 doc_id，跳过写入样例清单")
            continue
        records.append(
            {
                "id": doc_id,
                "file": relative_file,
                "subject": metadata["subject"],
                "grade": metadata["grade"],
                "chapter": metadata["chapter"],
            }
        )

    write_sample_manifest(paths, records)
    print(f"样例上传记录: {paths.sample_manifest}")
    return 0


def cmd_delete_samples(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    paths = RuntimePaths.from_root(root)
    base_url = build_base_url(root, args.url)
    deleted = delete_sample_documents(base_url, paths, HttpClient(timeout=args.timeout))
    print(f"已删除样例文档: {deleted}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    base_url = build_base_url(root, args.url)
    payload = {
        "query": args.query,
        "subject": args.subject,
        "grade": args.grade,
        "user_id": args.user_id,
        "session_id": args.session_id,
    }
    response = HttpClient(timeout=args.timeout).post_json(f"{base_url}/api/v1/rag/ask", payload)
    if args.json:
        print_json(response)
        return 0

    data = response.get("data") or {}
    print(data.get("answer") or response.get("message") or "")
    references = data.get("references") or []
    if references:
        print("\n引用:")
        for ref in references:
            print(f"- {ref.get('source', '')} score={ref.get('score', 0):.4f}")
    return 0 if response.get("code", 0) == 0 else 1


def cmd_test(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    python = resolve_python(root)
    commands = [
        [python, "test/test_cleaner.py", "--unit-only"],
        [python, "test/test_strategies.py", "--unit-only"],
        [python, "test/test_refactor_smoke.py"],
        [python, "test/test_project_cli.py"],
    ]
    for command in commands:
        print(f"\n运行: {' '.join(command)}")
        result = subprocess.run(command, cwd=root)
        if result.returncode != 0:
            return result.returncode
    return 0


def cmd_eval_sample(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    python = resolve_python(root)
    command = [python, "evaluation/cli.py", "validate", "--file", args.file]
    return subprocess.run(command, cwd=root).returncode


def cmd_auto_score(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    command = [
        resolve_python(root),
        "evaluation/cli.py",
        "evaluate-auto",
        "--limit",
        str(args.limit),
        "--name",
        args.name,
    ]
    if args.subject:
        command.extend(["--subject", args.subject])
    if args.grade:
        command.extend(["--grade", args.grade])
    if args.metrics:
        command.extend(["--metrics", args.metrics])
    if args.no_save:
        command.append("--no-save")
    return subprocess.run(command, cwd=root).returncode


def cmd_help(args: argparse.Namespace) -> int:
    """以普通子命令形式打印完整操作菜单，支持 `./edu-rag help`。"""
    build_parser().print_help()
    return 0


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default=str(PROJECT_ROOT), help="项目根目录，默认自动识别")


def add_service_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--url", default=None, help="服务地址，默认根据 .env 中 APP_PORT 生成")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP/等待超时时间，单位秒")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edu-rag",
        description="Edu-RAG 本地项目管理 CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_options(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    help_cmd = subparsers.add_parser("help", help="查看所有可用操作")
    help_cmd.set_defaults(func=cmd_help)

    setup = subparsers.add_parser("setup", help="首次初始化：创建 .env、虚拟环境并安装依赖")
    setup.add_argument("--skip-venv", action="store_true", help="跳过创建 .venv")
    setup.add_argument("--skip-install", action="store_true", help="跳过 pip install")
    setup.set_defaults(func=cmd_setup)

    start = subparsers.add_parser("start", help="后台启动服务")
    add_service_options(start)
    start.add_argument("--no-wait", dest="wait", action="store_false", help="启动后不等待健康检查")
    start.set_defaults(func=cmd_start, wait=True)

    stop = subparsers.add_parser("stop", help="停止服务")
    stop.add_argument("--timeout", type=int, default=15, help="等待进程退出的秒数")
    stop.add_argument("--force", action="store_true", help="超时后强制结束")
    stop.set_defaults(func=cmd_stop)

    restart = subparsers.add_parser("restart", help="重启服务")
    add_service_options(restart)
    restart.add_argument("--force", action="store_true", help="停止阶段超时后强制结束")
    restart.add_argument("--no-wait", dest="wait", action="store_false", help="启动后不等待健康检查")
    restart.set_defaults(func=cmd_restart, wait=True)

    status = subparsers.add_parser("status", help="查看本地进程状态")
    status.set_defaults(func=cmd_status)

    health = subparsers.add_parser("health", help="调用 /health")
    add_service_options(health)
    health.set_defaults(func=cmd_health)

    logs = subparsers.add_parser("logs", help="查看服务日志")
    logs.add_argument("--lines", type=int, default=80, help="显示最近多少行")
    logs.add_argument("-f", "--follow", action="store_true", help="持续跟随日志")
    logs.set_defaults(func=cmd_logs)

    open_cmd = subparsers.add_parser("open", help="打开 Web 控制台或接口文档")
    add_service_options(open_cmd)
    open_cmd.add_argument("--docs", action="store_true", help="打开 /docs")
    open_cmd.set_defaults(func=cmd_open)

    list_docs = subparsers.add_parser("list-docs", help="查看已上传文档")
    add_service_options(list_docs)
    list_docs.add_argument("--json", action="store_true", help="输出原始 JSON")
    list_docs.set_defaults(func=cmd_list_docs)

    upload_samples = subparsers.add_parser("upload-samples", help="上传 sample_docs 下的样例文档")
    add_service_options(upload_samples)
    upload_samples.add_argument("--subject", default=None, help="覆盖所有样例的学科；不传则从文件名推断")
    upload_samples.add_argument("--grade", default="", help="样例文档年级")
    upload_samples.add_argument("--strategy", default="recursive", help="切片策略")
    upload_samples.add_argument("--force", action="store_true", help="忽略已有样例清单，重新上传")
    upload_samples.set_defaults(func=cmd_upload_samples)

    delete_samples = subparsers.add_parser("delete-samples", help="删除由 upload-samples 上传的样例文档")
    add_service_options(delete_samples)
    delete_samples.set_defaults(func=cmd_delete_samples)

    ask = subparsers.add_parser("ask", help="向 RAG 服务提问")
    add_service_options(ask)
    ask.add_argument("query", help="问题内容")
    ask.add_argument("--subject", default=None, help="学科过滤")
    ask.add_argument("--grade", default=None, help="年级过滤")
    ask.add_argument("--user-id", default="cli-demo", help="用户 ID")
    ask.add_argument("--session-id", default=None, help="会话 ID")
    ask.add_argument("--json", action="store_true", help="输出原始 JSON")
    ask.set_defaults(func=cmd_ask)

    auto_score = subparsers.add_parser("as", help="评估最近自动沉淀的问答测试集")
    auto_score.add_argument("--limit", type=int, default=50, help="读取最近多少条自动样本")
    auto_score.add_argument("--subject", default=None, help="学科过滤")
    auto_score.add_argument("--grade", default=None, help="年级过滤")
    auto_score.add_argument("--metrics", default=None, help="评估指标，逗号分隔")
    auto_score.add_argument("--name", default="auto_samples", help="评估任务名称")
    auto_score.add_argument("--no-save", action="store_true", help="不保存评估结果到数据库")
    auto_score.set_defaults(func=cmd_auto_score)

    test = subparsers.add_parser("test", help="运行核心回归测试")
    test.set_defaults(func=cmd_test)

    eval_sample = subparsers.add_parser("eval-sample", help="校验随仓库提供的评估样例")
    eval_sample.add_argument("--file", default="data/test_sets/manual_v1.jsonl", help="测试集路径")
    eval_sample.set_defaults(func=cmd_eval_sample)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n已取消")
        return 130
    except Exception as exc:
        print(f"操作失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
