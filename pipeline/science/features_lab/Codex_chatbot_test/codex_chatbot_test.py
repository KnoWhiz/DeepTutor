#!/usr/bin/env python3
# minimal_codex_stream.py
import subprocess
import sys
from typing import Iterator, List, Optional

def codex_stream(project_dir: str, prompt: str, extra_args: Optional[List[str]] = None) -> Iterator[str]:
    """
    在 project_dir 中以非交互方式运行 `codex exec`，将 STDOUT/STDERR 按行流式产出。
    """
    args = ["codex", "exec"]
    if extra_args:
        args += extra_args
    args.append(prompt)

    # 关键点：使用 Popen + PIPE，并合并 stderr；bufsize=1 + text=True 便于逐行读取
    # （参见 Python 官方 subprocess 文档关于流式读取与 PIPE 的说明） [oai_citation:1‡Python documentation](https://docs.python.org/3/library/subprocess.html?utm_source=chatgpt.com)
    proc = subprocess.Popen(
        args,
        cwd=project_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            yield line
    finally:
        if proc.stdout:
            proc.stdout.close()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"codex exited with status {rc}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python minimal_codex_stream.py <项目文件夹> <提示语...>")
        sys.exit(1)

    project_dir = sys.argv[1]
    prompt = " ".join(sys.argv[2:])

    for chunk in codex_stream(project_dir, prompt):
        # 你也可以把 chunk 转发到 WebSocket/回调等
        print(chunk, end="")