"""Support code for Test code agent."""

from __future__ import annotations

import asyncio
import os
import sys
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from RL_Framework.env.code_agent_prompt import format_execution_result
from RL_Framework.env.code_executor import CodeExecutor, extract_code_block
from RL_Framework.env.code_reward import code_reward_fn
from RL_Framework.workflow.code_agent import CodeAgentWorkflow


# =============================================================================
# Test 1: Code block extraction
# =============================================================================


def test_extract_code_block():
    """Test extract code block."""
    print("\n[Test] extract_code_block")

    # Case 1: Standard python block
    text1 = 'Some reasoning...\n```python\nprint("hello")\n```\nMore text'
    code1 = extract_code_block(text1)
    assert code1 == 'print("hello")', f"Case 1 failed: {code1}"
    print("  Case 1 passed: standard python block")

    # Case 2: Generic code block
    text2 = 'Here is the code:\n```\nprint(42)\n```'
    code2 = extract_code_block(text2)
    assert code2 == "print(42)", f"Case 2 failed: {code2}"
    print("  Case 2 passed: generic code block")

    # Case 3: Block with language specifier
    text3 = '```python\ndef solve():\n    pass\n```'
    code3 = extract_code_block(text3)
    assert "def solve()" in code3, f"Case 3 failed: {code3}"
    print("  Case 3 passed: block with language specifier")

    # Case 4: No code block
    text4 = "Just some plain text without code"
    code4 = extract_code_block(text4)
    assert code4 is None, f"Case 4 failed: {code4}"
    print("  Case 4 passed: no code block")

    # Case 5: Multiple code blocks (should return last one)
    text5 = '```python\nfirst\n```\n```python\nsecond\n```'
    code5 = extract_code_block(text5)
    assert code5 == "second", f"Case 5 failed: {code5}"
    print("  Case 5 passed: multiple blocks (last one)")

    print("  All extract_code_block tests passed!")


# =============================================================================
# Test 2: Local sandbox execution
# =============================================================================


def test_local_sandbox_stdin():
    """Test local sandbox stdin."""
    print("\n[Test] local_sandbox_stdin")

    executor = CodeExecutor(mode="local", timeout=5)

    code = "import sys\nn = int(sys.stdin.readline())\nprint(n * 2)"
    test_cases = {
        "inputs": ["5\n", "10\n"],
        "outputs": ["10", "20"],
    }

    loop = asyncio.get_event_loop()
    pass_rate, metadata = loop.run_until_complete(executor.execute(code, test_cases))

    assert pass_rate == 1.0, f"Expected pass_rate=1.0, got {pass_rate}"
    assert metadata["passed"] == 2, f"Expected passed=2, got {metadata['passed']}"
    print(f"  pass_rate={pass_rate}, passed={metadata['passed']}/{metadata['total']}")
    print("  local_sandbox_stdin test passed!")


def test_local_sandbox_call_based():
    """Test local sandbox call based."""
    print("\n[Test] local_sandbox_call_based")

    executor = CodeExecutor(mode="local", timeout=5)

    code = """
class Solution:
    def add(self, a: int, b: int) -> int:
        return a + b
"""
    test_cases = {
        "fn_name": "add",
        "inputs": ["1\n2", "10\n20"],
        "outputs": ["3", "30"],
    }

    loop = asyncio.get_event_loop()
    pass_rate, metadata = loop.run_until_complete(executor.execute(code, test_cases))

    assert pass_rate == 1.0, f"Expected pass_rate=1.0, got {pass_rate}"
    print(f"  pass_rate={pass_rate}, passed={metadata['passed']}/{metadata['total']}")
    print("  local_sandbox_call_based test passed!")


def test_local_sandbox_failure():
    """Test local sandbox failure."""
    print("\n[Test] local_sandbox_failure")

    executor = CodeExecutor(mode="local", timeout=5)

    # Wrong output
    code = "import sys\nn = int(sys.stdin.readline())\nprint(n * 3)"  # should be * 2
    test_cases = {
        "inputs": ["5\n"],
        "outputs": ["10"],
    }

    loop = asyncio.get_event_loop()
    pass_rate, metadata = loop.run_until_complete(executor.execute(code, test_cases))

    assert pass_rate == 0.0, f"Expected pass_rate=0.0, got {pass_rate}"
    print(f"  pass_rate={pass_rate} (expected 0.0 for wrong answer)")
    print("  local_sandbox_failure test passed!")


# =============================================================================
# Test 3: Reward function
# =============================================================================


def test_code_reward_fn():
    """Test code reward fn."""
    print("\n[Test] code_reward_fn")

    executor = CodeExecutor(mode="local", timeout=5)

    # Correct code
    code_correct = '```python\nimport sys\nn = int(sys.stdin.readline())\nprint(n * 2)\n```'
    test_cases = {
        "inputs": ["5\n"],
        "outputs": ["10"],
    }

    reward = code_reward_fn(
        prompt="Solve the problem",
        completion=code_correct,
        test_cases=test_cases,
        executor=executor,
        continuous=True,
        format_weight=0.1,
    )
    assert reward > 0.9, f"Expected reward > 0.9 for correct code, got {reward}"
    print(f"  Correct code reward: {reward:.3f}")

    # Wrong code
    code_wrong = '```python\nimport sys\nn = int(sys.stdin.readline())\nprint(n * 3)\n```'
    reward_wrong = code_reward_fn(
        prompt="Solve the problem",
        completion=code_wrong,
        test_cases=test_cases,
        executor=executor,
        continuous=True,
        format_weight=0.1,
    )
    # Should get format_weight only (0.1) since code is wrong
    assert reward_wrong < 0.2, f"Expected reward < 0.2 for wrong code, got {reward_wrong}"
    print(f"  Wrong code reward: {reward_wrong:.3f}")

    # No code block
    code_none = "I think the answer is 10"
    reward_none = code_reward_fn(
        prompt="Solve the problem",
        completion=code_none,
        test_cases=test_cases,
        executor=executor,
        continuous=True,
        format_weight=0.1,
    )
    assert reward_none == 0.1, f"Expected reward=0.1 (format only), got {reward_none}"
    print(f"  No code block reward: {reward_none:.3f}")

    print("  code_reward_fn tests passed!")


# =============================================================================
# Test 4: Execution result formatting
# =============================================================================


def test_format_execution_result():
    """Test format execution result."""
    print("\n[Test] format_execution_result")

    # All passed
    result1 = format_execution_result(
        passed=3, total=3,
        metadata_list=[{"status": "success"}, {"status": "success"}, {"status": "success"}]
    )
    assert "3/3" in result1, "Should show 3/3 passed"
    assert "correct" in result1.lower(), "Should mention correct"
    print("  All passed formatting OK")

    # Partial
    result2 = format_execution_result(
        passed=1, total=3,
        metadata_list=[
            {"status": "success"},
            {"status": "wrong_answer", "input": "5", "expected_output": "10", "stdout": "15"},
            {"status": "runtime_error", "error": "IndexError"},
        ]
    )
    assert "1/3" in result2, "Should show 1/3 passed"
    assert "Failed Cases" in result2, "Should list failed cases"
    print("  Partial pass formatting OK")

    # All failed
    result3 = format_execution_result(
        passed=0, total=2,
        metadata_list=[
            {"status": "wrong_answer", "input": "1", "expected_output": "2", "stdout": "3"},
            {"status": "wrong_answer", "input": "2", "expected_output": "4", "stdout": "5"},
        ]
    )
    assert "0/2" in result3 or "All tests failed" in result3, "Should show all failed"
    print("  All failed formatting OK")

    print("  format_execution_result tests passed!")


# =============================================================================
# Test 5: CodeAgentWorkflow with mock engine
# =============================================================================


class MockEngine:
    """Mock engine implementation."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or []
        self.call_count = 0

    async def generate(self, prompt, max_new_tokens, temperature, n):
        if self.call_count < len(self.responses):
            text = self.responses[self.call_count]
        else:
            text = "```python\nprint(1)\n```"
        self.call_count += 1

        tokens = text.split()  # rough tokenization for testing
        logprobs = [-0.1] * len(tokens)
        return {"text": text, "logprobs": logprobs}


def test_code_agent_workflow():
    """Test code agent workflow."""
    print("\n[Test] CodeAgentWorkflow")

    from transformers import AutoTokenizer


    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        print("  Skipping workflow test (gpt2 tokenizer not available)")
        return

    executor = CodeExecutor(mode="local", timeout=5)

    def simple_reward_fn(prompt, completion, **kwargs):
        # Simple reward for testing
        if "```python" in completion:
            return 0.5
        return 0.0

    workflow = CodeAgentWorkflow(
        reward_fn=simple_reward_fn,
        tokenizer=tokenizer,
        executor=executor,
        max_turns=3,
        max_new_tokens=100,
        temperature=1.0,
        stop_on_submit=True,
    )

    # Mock engine: first response has code, second has submit tag
    mock_engine = MockEngine(responses=[
        '```python\nimport sys\nn = int(sys.stdin.readline())\nprint(n * 2)\n```',
        '<submit>',
    ])

    data = {
        "question": "Read an integer and print its double.",
        "test_cases": {
            "inputs": ["5\n"],
            "outputs": ["10"],
        },
    }

    loop = asyncio.get_event_loop()
    trajectory = loop.run_until_complete(workflow.run_episode(mock_engine, data))

    # Check trajectory structure
    assert "input_ids" in trajectory, "Trajectory should have input_ids"
    assert "rewards" in trajectory, "Trajectory should have rewards"
    assert "loss_mask" in trajectory, "Trajectory should have loss_mask"
    assert trajectory["rewards"].item() == 0.5, f"Expected reward=0.5, got {trajectory['rewards'].item()}"

    # Check that loss_mask has both 0 and 1 values
    loss_mask = trajectory["loss_mask"].squeeze().tolist()
    assert 0 in loss_mask, "loss_mask should have 0s (prompt/tool parts)"
    assert 1 in loss_mask, "loss_mask should have 1s (generation parts)"

    print(f"  Reward: {trajectory['rewards'].item()}")
    print(f"  n_turns: {trajectory['n_turns']}")
    print(f"  n_executions: {trajectory['n_executions']}")
    print(f"  Seq length: {trajectory['input_ids'].shape[1]}")
    print("  CodeAgentWorkflow test passed!")


# =============================================================================
# Main test runner
# =============================================================================


def run_all_tests():
    print("=" * 60)
    print("CodeAgent module unit tests")
    print("=" * 60)

    test_extract_code_block()
    test_local_sandbox_stdin()
    test_local_sandbox_call_based()
    test_local_sandbox_failure()
    test_code_reward_fn()
    test_format_execution_result()
    test_code_agent_workflow()

    print("\n" + "=" * 60)
    print("All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
