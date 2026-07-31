"""Support code for Code reward."""

from __future__ import annotations

import logging
from typing import Any, Optional

from env.code_executor import CodeExecutor

logger = logging.getLogger(__name__)


# =============================================================================
# Core reward functions
# =============================================================================


def extract_code_block(text: str) -> Optional[str]:
    """Extract code block."""
    if "```python" in text:
        code = text.split("```python")[-1].split("```")[0]
        return code.strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            code = parts[1]
            if "\n" in code:
                first_line, rest = code.split("\n", 1)
                if first_line.strip().isalpha():
                    return rest.strip()
            return code.strip()
    return None


def code_reward_fn(
    prompt: str,
    completion: str,
    test_cases: dict[str, Any],
    executor: Optional[CodeExecutor] = None,
    continuous: bool = True,
    format_weight: float = 0.0,
    **kwargs,
) -> float:
    """Code reward fn."""
    del prompt  # unused


    code = extract_code_block(completion)
    if not code:

        return format_weight


    if executor is None:
        executor = CodeExecutor(mode="local", timeout=10)


    try:
        import asyncio

        loop = asyncio.get_event_loop()
        if loop.is_running():


            pass_rate, metadata = asyncio.run(executor.execute(code, test_cases))
        else:
            pass_rate, metadata = asyncio.run(executor.execute(code, test_cases))
    except RuntimeError:
        # Event loop already running (e.g., inside Jupyter or nested async)
        # Fallback: create a new loop in a thread
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, executor.execute(code, test_cases))
            pass_rate, metadata = future.result()
    except Exception as e:
        logger.warning(f"Code execution failed: {e}")
        pass_rate = 0.0
        metadata = {"error": str(e)}


    if continuous:

        reward = pass_rate * (1.0 - format_weight) + format_weight
    else:

        reward = 1.0 if pass_rate >= 1.0 - 1e-6 else 0.0
        if format_weight > 0:
            reward = reward * (1.0 - format_weight) + format_weight

    logger.debug(
        f"code_reward: pass_rate={pass_rate:.3f}, continuous={continuous}, "
        f"format_weight={format_weight}, reward={reward:.3f}"
    )
    return reward


def combined_code_reward_fn(
    prompt: str,
    completion: str,
    test_cases: dict[str, Any],
    executor: Optional[CodeExecutor] = None,
    format_weight: float = 0.1,
    compile_bonus: float = 0.0,
    **kwargs,
) -> float:
    """Combined code reward fn."""

    base_reward = code_reward_fn(
        prompt=prompt,
        completion=completion,
        test_cases=test_cases,
        executor=executor,
        continuous=True,
        format_weight=format_weight,
        **kwargs,
    )


    if compile_bonus > 0.0:
        code = extract_code_block(completion)
        if code:

            try:
                import ast

                ast.parse(code)
                base_reward += compile_bonus
            except SyntaxError:
                pass

    return min(base_reward, 1.0)


# =============================================================================
# Compatibility wrapper for RL_Framework trainer
# =============================================================================


def make_code_reward_fn(
    executor: Optional[CodeExecutor] = None,
    continuous: bool = True,
    format_weight: float = 0.1,
    compile_bonus: float = 0.0,
):
    """Make code reward fn."""
    _executor = executor or CodeExecutor(mode="local", timeout=10)

    def _reward_fn(prompt: str, completion: str, **kwargs) -> float:
        test_cases = kwargs.get("test_cases")
        if not test_cases:
            logger.warning("No test_cases provided to code_reward_fn, returning 0.0")
            return 0.0

        return combined_code_reward_fn(
            prompt=prompt,
            completion=completion,
            test_cases=test_cases,
            executor=_executor,
            format_weight=format_weight,
            compile_bonus=compile_bonus,
        )

    return _reward_fn
