"""Support code for Verify code agent."""

from __future__ import annotations

import ast
import asyncio
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def check_syntax(filepath: str) -> bool:
    with open(filepath) as f:
        source = f.read()
    try:
        ast.parse(source)
        return True
    except SyntaxError as e:
        print(f"  SyntaxError: {e}")
        return False


def main():
    print("=" * 60)
    print("CodeAgent module verification")
    print("=" * 60)

    # ---- 1. Syntax check all new files ----
    print("\n[1/7] Syntax check...")
    files = [
        "env/code_executor.py",
        "env/code_agent_prompt.py",
        "env/code_reward.py",
        "workflow/code_agent.py",
        "data/preprocess_livecodebench.py",
        "examples/livecodebench_code_agent.py",
    ]
    all_ok = True
    for f in files:
        ok = check_syntax(_PROJECT_ROOT / f)
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {f}")
        all_ok = all_ok and ok
    if not all_ok:
        sys.exit(1)

    # ---- 2. Import check ----
    print("\n[2/7] Module imports...")
    from env.code_executor import CodeExecutor
    from env.code_agent_prompt import format_execution_result, format_problem
    from env.code_reward import extract_code_block
    print("  OK: All core modules imported")

    # ---- 3. Code block extraction ----
    print("\n[3/7] Code block extraction...")
    assert extract_code_block("```python\nprint(1)\n```") == "print(1)"
    assert extract_code_block("no code") is None
    assert extract_code_block("```\ncode\n```") == "code"
    print("  OK")

    # ---- 4. Local sandbox (stdin mode) ----
    print("\n[4/7] Local sandbox (stdin mode)...")
    executor = CodeExecutor(mode="local", timeout=5)
    code = "import sys\nn = int(sys.stdin.readline())\nprint(n * 2)"
    test_cases = {"inputs": ["5\n"], "outputs": ["10"]}
    loop = asyncio.get_event_loop()
    pass_rate, metadata = loop.run_until_complete(executor.execute(code, test_cases))
    assert pass_rate == 1.0, f"Expected 1.0, got {pass_rate}"
    print(f"  OK: pass_rate={pass_rate}")

    # ---- 5. Local sandbox (call-based) ----
    print("\n[5/7] Local sandbox (call-based)...")
    code2 = "\nclass Solution:\n    def add(self, a, b):\n        return a + b\n"
    test_cases2 = {"fn_name": "add", "inputs": ["1\n2"], "outputs": ["3"]}
    pass_rate2, _ = loop.run_until_complete(executor.execute(code2, test_cases2))
    assert pass_rate2 == 1.0, f"Expected 1.0, got {pass_rate2}"
    print(f"  OK: pass_rate={pass_rate2}")

    # ---- 6. Local sandbox (wrong answer) ----
    print("\n[6/7] Local sandbox (wrong answer)...")
    code3 = "import sys\nn = int(sys.stdin.readline())\nprint(n * 3)"
    pass_rate3, _ = loop.run_until_complete(executor.execute(code3, test_cases))
    assert pass_rate3 == 0.0, f"Expected 0.0, got {pass_rate3}"
    print(f"  OK: pass_rate={pass_rate3} (expected failure)")

    # ---- 7. Prompt formatting ----
    print("\n[7/7] Prompt formatting...")
    result = format_execution_result(
        2, 3,
        [
            {"status": "success"},
            {"status": "wrong_answer", "input": "1", "expected_output": "2", "stdout": "3"},
            {"status": "success"},
        ]
    )
    assert "2/3" in result
    problem = format_problem("Read N and print N*2", "def solve():\n    pass")
    assert "Read N" in problem
    assert "Starter Code" in problem
    print("  OK")

    print("\n" + "=" * 60)
    print("All checks passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
