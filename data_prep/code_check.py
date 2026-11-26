"""Run a candidate Python solution against a problem's tests in a throwaway process (post-training data: the code
problem pool and the checks on teacher-written code).

    ok, detail = check(code, spec)

spec is one of
    {"type": "asserts", "code": "assert f(1) == 2\\n...", "setup": ""}       the tests run after the solution
    {"type": "stdin", "cases": [[stdin text, expected stdout], ...]}           whitespace-separated tokens must match
    {"type": "call", "fn": "name", "cases": [[args list, expected], ...]}      f(*args) or Solution().f(*args)

The process runs `python -I` in an empty temporary directory with a minimal environment, a time limit and limits on
CPU time, memory, file size and processes; as root (WSL) it also drops to the user nobody, so it cannot write outside
that directory. Code that matches UNSAFE (process, file-system and network modules) is not run at all. This guards
against accidents in generated code; it is not a security boundary for hostile code.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time

UNSAFE = re.compile(r"\b(os|subprocess|shutil|socket|pathlib|ctypes|importlib|signal|multiprocessing|threading|concurrent|"
                    r"requests|urllib|http|ftplib|pickle|marshal|pty|resource)\b"
                    r"|__import__|\bopen\s*\(|\beval\s*\(|\bexec\s*\(|\bcompile\s*\(|rmtree|unlink|sys\.modules"
                    r"|__builtins__|__subclasses__|breakpoint\s*\(")
PRELUDE = ("import sys\nsys.setrecursionlimit(100000)\nsys.set_int_max_str_digits(0)\n"
           "from typing import *\nimport math, collections, itertools, functools, heapq, bisect, re, string\n")
_CALL = r'''
import json as _json
_cases = _json.loads({cases!r})
_fn = getattr(Solution(), {fn!r}) if "Solution" in dir() else globals()[{fn!r}]
def _same(a, b):
    if isinstance(a, tuple):
        a = list(a)
    if a == b or [a] == b:
        return True
    try:
        return abs(float(a) - float(b[0] if isinstance(b, list) and len(b) == 1 else b)) < 1e-6
    except Exception:
        return False
for _i, (_args, _exp) in enumerate(_cases):
    _out = _fn(*_args)
    assert _same(_out, _exp), "case %d: got %r, expected %r" % (_i, _out, _exp)
'''


def _limits():
    # become nobody first and only then lower the limits: with RLIMIT_NPROC already set, a setuid() made while the
    # user has that many processes marks the process, and its execve() then fails with EAGAIN (BlockingIOError)
    if os.geteuid() == 0:
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)
    import resource
    for what, lim in ((resource.RLIMIT_CPU, 20), (resource.RLIMIT_FSIZE, 1 << 20), (resource.RLIMIT_NPROC, 256),
                      (resource.RLIMIT_AS, 2 << 30)):
        try:
            resource.setrlimit(what, (lim, lim))
        except (ValueError, OSError):
            pass  # macOS refuses some of these


def _python() -> str:
    """The interpreter for candidate code: as root the system one (the venv under /root is closed to nobody)."""
    if os.geteuid() == 0 and os.access("/usr/bin/python3", os.X_OK):
        return "/usr/bin/python3"
    return sys.executable


def _kill_group(proc):
    """Everything the candidate started (it may start processes of its own) dies with it."""
    import signal
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run(code: str, stdin: str = "", timeout: float = 10, files: dict | None = None, main: str = "main.py",
        err_chars: int = 300):
    """(exit code 0, stdout, last 300 characters of stderr); ("timeout" as stderr when the time limit hit).
    files: other files {name: text} written next to the script (a small project: data_prep/agent_env.py); main: the
    script's own file name, which tracebacks show."""
    # as root the default TMPDIR may sit under /root, which the user nobody cannot enter
    with tempfile.TemporaryDirectory(dir="/tmp" if os.geteuid() == 0 else None) as d:
        os.chmod(d, 0o777)
        for name, text in (files or {}).items():
            fp = os.path.join(d, os.path.basename(name))
            with open(fp, "w", encoding="utf-8") as f:
                f.write(text)
            os.chmod(fp, 0o644)
        path = os.path.join(d, os.path.basename(main))  # a file, not -c: one argument is limited to 128 KB
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        os.chmod(path, 0o644)
        for attempt in range(6):
            try:
                # -I implies -P since Python 3.11: the script's directory is then not on sys.path, and a project's test file
                # could not import its solution.py; -E -s keep the rest of the isolation (no PYTHON* variables, no user site)
                flags = ["-E", "-s"] if files else ["-I"]
                proc = subprocess.Popen([_python(), *flags, path], cwd=d, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, env={"PATH": "/usr/bin:/bin", "HOME": d},
                                        preexec_fn=_limits, start_new_session=True)
            except BlockingIOError as e:  # EAGAIN: the machine is out of processes for a moment, try again
                if attempt == 5:
                    return False, "", f"could not run: {e!r}"[:300]
                time.sleep(0.2 * 2 ** attempt)
                continue
            except (OSError, UnicodeError, subprocess.SubprocessError) as e:
                return False, "", f"could not run: {e!r}"[:300]
            try:
                out, err = proc.communicate(input=stdin, timeout=timeout)
                return proc.returncode == 0, out, err[-err_chars:]
            except subprocess.TimeoutExpired:
                return False, "", "timeout"
            except (OSError, UnicodeError, subprocess.SubprocessError) as e:
                return False, "", f"could not run: {e!r}"[:300]
            finally:
                _kill_group(proc)
                try:
                    proc.communicate(timeout=5)
                except Exception:  # noqa: BLE001  already gone, or stuck: nothing more to do
                    pass


def _tokens_match(a: str, b: str) -> bool:
    ta, tb = a.split(), b.split()
    if ta == tb:
        return True
    if len(ta) != len(tb):
        return False
    for x, y in zip(ta, tb):
        if x != y:
            try:
                if abs(float(x) - float(y)) > 1e-6:
                    return False
            except ValueError:
                return False
    return True


def check(code: str, spec: dict, timeout: float = 10):
    """(passed, detail). detail names the first failure."""
    if UNSAFE.search(code):
        return False, "not run (unsafe pattern)"
    kind = spec["type"]
    if kind == "asserts":
        ok, _, err = run(PRELUDE + code + "\n" + (spec.get("setup") or "") + "\n" + spec["code"] + "\n", timeout=timeout)
        return ok, err
    if kind == "call":
        harness = _CALL.format(cases=json.dumps(spec["cases"]), fn=spec["fn"])
        ok, _, err = run(PRELUDE + code + "\n" + harness, timeout=timeout)
        return ok, err
    if kind == "stdin":
        for i, (inp, want) in enumerate(spec["cases"]):
            ok, out, err = run(code, stdin=inp, timeout=timeout)
            if not ok:
                return False, f"case {i}: {err}"
            if not _tokens_match(out, want):
                return False, f"case {i}: wrong output"
        return True, ""
    raise ValueError(kind)


def code_block(text: str):
    """The last ```python (or bare ```) block of a reply, or None."""
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.S)
    return blocks[-1] if blocks else None
