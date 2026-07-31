"""Support code for Code agent prompt."""

# =============================================================================
# System Prompt
# =============================================================================

CODE_AGENT_SYSTEM_PROMPT = """You are an expert competitive programming assistant. Your task is to solve programming problems by writing Python code.

## Instructions:
1. Read the problem description carefully.
2. Think step by step about the solution approach.
3. Write clean, correct, and efficient Python code.
4. Your code must be wrapped in a ```python code block.
5. After writing code, you can see the execution results and fix errors if needed.
6. When you believe your solution is correct, output <submit> to finish.

## Rules:
- Use standard input/output (sys.stdin / print) unless the problem specifies a function signature.
- Do NOT write test cases or example usage in your code.
- Do NOT include any explanatory text outside the code block, except for brief reasoning before the code.
- Make sure your code handles all edge cases correctly.

## Format:
Your response should follow this format:
```python
# Your solution code here
```
"""

# =============================================================================
# User problem format template
# =============================================================================

USER_PROBLEM_FORMAT = """### Problem
{question}

{starter_code_block}

Please solve this problem in Python. Wrap your code in ```python ... ```."""

STARTER_CODE_BLOCK = """### Starter Code
```python
{starter_code}
```"""

# =============================================================================
# Execution result injection template (used as tool response)
# =============================================================================

CODE_EXECUTION_RESULT_TEMPLATE = """### Execution Result

{status_line}

{details}
"""

EXECUTION_STATUS_PASSED = "All {passed}/{total} test cases passed!"
EXECUTION_STATUS_PARTIAL = "{passed}/{total} test cases passed. Some tests failed."
EXECUTION_STATUS_FAILED = "All tests failed."
EXECUTION_STATUS_ERROR = "Execution error: {error_message}"

EXECUTION_DETAILS_TEMPLATE = """**Failed Cases:**
{failed_cases}

Please analyze the errors and provide a corrected solution.
"""

FAILED_CASE_TEMPLATE = """- Case {case_index}: {status}
  Input: `{input_data}`
  Expected: `{expected_output}`
  Actual: `{actual_output}`
  Error: `{error_message}`
"""

# =============================================================================
# Helper functions to format prompts
# =============================================================================


def format_problem(question: str, starter_code: str | None = None) -> str:
    """Format problem."""
    if starter_code and starter_code.strip():
        starter_block = STARTER_CODE_BLOCK.format(starter_code=starter_code)
    else:
        starter_block = ""
    return USER_PROBLEM_FORMAT.format(
        question=question,
        starter_code_block=starter_block,
    )


def format_execution_result(
    passed: int,
    total: int,
    metadata_list: list[dict],
    max_failed_cases: int = 3,
) -> str:
    """Format execution result."""
    if passed == total and total > 0:
        status_line = EXECUTION_STATUS_PASSED.format(passed=passed, total=total)
        return CODE_EXECUTION_RESULT_TEMPLATE.format(
            status_line=status_line,
            details="Your solution is correct! You can now submit.",
        )

    if total == 0:
        status_line = EXECUTION_STATUS_ERROR.format(error_message="No test cases available")
        return CODE_EXECUTION_RESULT_TEMPLATE.format(status_line=status_line, details="")

    if passed == 0:
        status_line = EXECUTION_STATUS_FAILED
    else:
        status_line = EXECUTION_STATUS_PARTIAL.format(passed=passed, total=total)

    # Collect failed case details
    failed_cases = []
    for i, meta in enumerate(metadata_list):
        status = meta.get("status", "unknown")
        if status == "success":
            continue

        input_data = meta.get("input", "N/A")
        expected = meta.get("expected_output", "N/A")
        actual = meta.get("stdout", meta.get("output", "N/A"))
        error = meta.get("error", meta.get("error_message", status))

        # Truncate long strings
        if isinstance(input_data, str) and len(input_data) > 200:
            input_data = input_data[:200] + "..."
        if isinstance(expected, str) and len(expected) > 200:
            expected = expected[:200] + "..."
        if isinstance(actual, str) and len(actual) > 200:
            actual = actual[:200] + "..."

        failed_cases.append(
            FAILED_CASE_TEMPLATE.format(
                case_index=i + 1,
                status=status,
                input_data=input_data,
                expected_output=expected,
                actual_output=actual,
                error_message=error,
            )
        )

        if len(failed_cases) >= max_failed_cases:
            remaining = sum(1 for m in metadata_list[i + 1 :] if m.get("status") != "success")
            if remaining > 0:
                failed_cases.append(f"\n... and {remaining} more failed cases.")
            break

    details = EXECUTION_DETAILS_TEMPLATE.format(failed_cases="".join(failed_cases))
    return CODE_EXECUTION_RESULT_TEMPLATE.format(status_line=status_line, details=details)


# =============================================================================
# Conversation role markers for internal tracking
# =============================================================================

# These are NOT sent to the model directly; they are used by the workflow
# to construct the conversation history before apply_chat_template.

TOOL_ROLE_MARKER = "tool"
ASSISTANT_ROLE_MARKER = "assistant"
USER_ROLE_MARKER = "user"

# End-of-episode markers
SUBMIT_TAG = "<submit>"
STOP_TAG = "<stop>"
