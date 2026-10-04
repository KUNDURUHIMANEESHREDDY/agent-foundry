"""Direct verification of python.execute claims. Not a substitute for the eval
suite — this checks the subprocess behaviours the scripted cases cannot see.

PYTHONPATH=src python verify_python_exec.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

from factory.capabilities.python_exec import PythonExecute

WS = Path("evals/workspace").resolve()
cap = PythonExecute(workspace=WS, timeout_s=2.0, max_output_bytes=4096)
failures = []


def check(label, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {label}{(' — ' + detail) if detail else ''}")
    if not cond:
        failures.append(label)


print("\n=== timeout kills the whole process tree ===")
marker = WS / "orphan.txt"
marker.unlink(missing_ok=True)

# Written to files rather than embedded as escaped strings, so the code under
# test is unambiguous.
child_src = WS / "_child.py"
child_src.write_text(
    "import time\n"
    f"f = open({str(marker)!r}, 'a')\n"
    "for _ in range(200):\n"
    "    f.write('x'); f.flush(); time.sleep(0.2)\n",
    encoding="utf-8",
)
parent_src = WS / "_parent.py"
parent_src.write_text(
    "import subprocess, sys, time\n"
    f"subprocess.Popen([sys.executable, {str(child_src)!r}])\n"
    "time.sleep(60)\n",
    encoding="utf-8",
)

start = time.monotonic()
res = cap.invoke(code=f"exec(open({str(parent_src)!r}).read())")
elapsed = time.monotonic() - start

check("timed_out reported", res["timed_out"] is True, f"elapsed={elapsed:.1f}s exit={res['exit_code']}")
check("returned near the 2s timeout, not 60s", elapsed < 12, f"{elapsed:.1f}s")
time.sleep(2.0)
size_after = marker.stat().st_size if marker.exists() else 0
time.sleep(2.5)
size_later = marker.stat().st_size if marker.exists() else 0
check("grandchild stopped writing (tree killed)", size_after == size_later,
      f"{size_after} -> {size_later} bytes")
check("grandchild did write at all (test is meaningful)", size_after > 0,
      f"{size_after} bytes written before kill")

for p in (marker, child_src, parent_src):
    p.unlink(missing_ok=True)

print("\n=== output is capped in memory, not just in the observation ===")
huge = 40 * 1024 * 1024  # 40MB, well past the 4KB cap
res = cap.invoke(code=f"print('z'*{huge})")
check("truncated flag set", res["truncated"] is True)
check("stored stdout <= cap", len(res["stdout"]) <= 4096, f"{len(res['stdout'])} chars")
check("true byte count reported", res["stdout_total_bytes"] >= huge,
      f"reported {res['stdout_total_bytes']}")
check("did not hang on full pipe", res["duration_ms"] < 15000, f"{res['duration_ms']}ms")

print("\n=== environment is scrubbed ===")
os.environ["FACTORY_VERIFY_SECRET"] = "leaked-value-here"
res = cap.invoke(code="import os; print(os.environ.get('FACTORY_VERIFY_SECRET', 'ABSENT'))")
check("secret not visible to child", "leaked-value-here" not in res["stdout"],
      res["stdout"].strip()[:40])
res = cap.invoke(code="import os; print(sorted(os.environ.keys()))")
check("env is small", res["stdout"].count(",") < 40, f"{len(res['stdout'])} bytes")
os.environ.pop("FACTORY_VERIFY_SECRET", None)

print("\n=== cwd is pinned, stdin is closed ===")
res = cap.invoke(code="import os; print(os.getcwd())")
check("cwd is workspace", "workspace" in res["stdout"], res["stdout"].strip())
res = cap.invoke(code="print('stdin is', 'closed' if __import__('sys').stdin.read() == '' else 'open')")
check("stdin is not a tty / is empty", "closed" in res["stdout"] or res["timed_out"],
      res["stdout"].strip()[:40] or "timed out reading stdin")

print("\n=== documented limitation is real (NOT a safety guarantee) ===")
res = cap.invoke(code="print(open('../secret.txt').read()[:40])")
check("subprocess CAN read outside workspace (expected)", res["ok"] is True,
      res["stdout"].strip()[:40])
print("     ^ this is the documented limitation, asserted so it cannot be lost")

print("\n=== interpreter is isolated (-I) ===")
res = cap.invoke(code="import sys; print(sys.flags.isolated)")
check("isolated mode on", res["stdout"].strip() == "1", res["stdout"].strip())

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    raise SystemExit(1)
print("all python.execute claims verified")
