"""data_prep/code_check.py: running candidate solutions against tests."""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_prep"))
import code_check as C  # noqa: E402


def test_asserts():
    spec = {"type": "asserts", "code": "assert add(1, 2) == 3\nassert add(-1, 1) == 0"}
    assert C.check("def add(a, b):\n    return a + b", spec)[0]
    ok, err = C.check("def add(a, b):\n    return a - b", spec)
    assert not ok and "AssertionError" in err
    assert C.check("def add(a: List[int], b):\n    return 3 if b == 2 else 0", spec)[0]  # typing names are imported


def test_stdin():
    spec = {"type": "stdin", "cases": [["3\n1 2 3\n", "6\n"], ["2\n0.5 0.25\n", "0.75"]]}
    code = "n = int(input())\nprint(sum(map(float, input().split())) if n == 2 else sum(map(int, input().split())))"
    assert C.check(code, spec)[0]
    assert C.check("import sys\nprint(sum(map(float, sys.stdin.read().split()[1:])))", spec)[0]  # 6.0 == 6
    assert C.check("print(7)", spec) == (False, "case 0: wrong output")


def test_call():
    spec = {"type": "call", "fn": "maxScore", "cases": [[[[1, 2, 3], 2], 5], [[[4], 1], [4]]]}
    sol = "class Solution:\n    def maxScore(self, xs: List[int], k: int) -> int:\n        return sum(sorted(xs)[-k:])"
    assert C.check(sol, spec)[0]
    assert C.check("def maxScore(xs, k):\n    return sum(sorted(xs)[-k:])", spec)[0]
    ok, err = C.check("def maxScore(xs, k):\n    return 0", spec)
    assert not ok and "case 0" in err


def test_guards():
    spec = {"type": "asserts", "code": "assert f() == 1"}
    assert C.check("import os\ndef f():\n    return 1", spec) == (False, "not run (unsafe pattern)")
    assert C.check("def f():\n    return open('x').read()", spec)[1] == "not run (unsafe pattern)"
    ok, err = C.check("def f():\n    while True:\n        pass", spec, timeout=2)
    assert not ok and (err == "timeout" or err)
    assert C.code_block("text\n```python\ndef f():\n    return 1\n```\nmore\n```\nx = 2\n```") == "x = 2\n"
    assert C.code_block("no code") is None


def test_process_pools_are_not_run():
    # a solution that starts worker processes (concurrent.futures re-runs the file in every worker) is not run: such a
    # chain of processes used up the process limit and made other checks fail with BlockingIOError
    spec = {"type": "asserts", "code": "assert f() == 1"}
    code = "from concurrent.futures import ProcessPoolExecutor\ndef f():\n    return 1"
    assert C.check(code, spec) == (False, "not run (unsafe pattern)")


def test_many_checks_at_once():
    # 24 checks in parallel threads all run (no spurious "could not run"), pass and fail as they should
    import concurrent.futures as cf
    spec = {"type": "asserts", "code": "assert f(2) == 4"}
    good, bad = "def f(x):\n    return x * x", "def f(x):\n    return x + 1"
    with cf.ThreadPoolExecutor(24) as ex:
        res = list(ex.map(lambda i: C.check(good if i % 2 == 0 else bad, spec), range(48)))
    assert all(r[0] for r in res[0::2])
    assert all(not r[0] and "AssertionError" in r[1] for r in res[1::2])


def test_timeout_kills_the_group(tmp_path):
    # the candidate starts a child that outlives it; when the time limit hits, the child must be killed too
    marker = str(tmp_path / "alive")
    code = ("import subprocess, sys, time\n"
            f"subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)', '{marker}'])\n"
            "time.sleep(30)\n")
    ok, _, err = C.run(code, timeout=1.5)
    assert not ok and err == "timeout"
    time.sleep(0.5)
    left = subprocess.run(["pgrep", "-f", f"time.sleep\\(30\\)', '{marker}"], capture_output=True, text=True).stdout.split()
    assert left == []
