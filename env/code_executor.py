"""Support code for Code executor."""

from __future__ import annotations

import ast
import asyncio
import concurrent.futures
import faulthandler
import json
import logging
import multiprocessing
import os
import platform
import re
import signal
import sys
import threading
import time
import traceback
import uuid
from io import StringIO
from typing import Any, Optional
from unittest.mock import mock_open, patch

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional pyext import (used by verl prime_code for RuntimeModule)
# ---------------------------------------------------------------------------
try:
    from pyext import RuntimeModule

    _HAS_PYEXT = True
except ImportError:
    _HAS_PYEXT = False
    RuntimeModule = None  # type: ignore

# ---------------------------------------------------------------------------
# Optional requests import (used by remote sandbox_fusion mode)
# ---------------------------------------------------------------------------
try:
    import requests

    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TIMEOUT = 10
DEFAULT_MEMORY_LIMIT_MB = 1024
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1
API_TIMEOUT = 10

SUPPORTED_LANGUAGES = [
    "python",
    "cpp",
    "nodejs",
    "go",
    "java",
    "php",
    "csharp",
    "rust",
    "kotlin_script",
]

STANDARD_IMPORTS = """from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json
sys.setrecursionlimit(6*10**5)
"""


# =============================================================================
# Helper: reliability_guard (disable dangerous builtins)
# =============================================================================


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """Reliability guard."""
    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if platform.uname().system != "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os as _os

    _os.environ["OMP_NUM_THREADS"] = "1"
    _os.kill = None
    _os.system = None
    _os.putenv = None
    _os.remove = None
    _os.removedirs = None
    _os.rmdir = None
    _os.fchdir = None
    _os.setuid = None
    _os.fork = None
    _os.forkpty = None
    _os.killpg = None
    _os.rename = None
    _os.renames = None
    _os.truncate = None
    _os.replace = None
    _os.unlink = None
    _os.fchmod = None
    _os.fchown = None
    _os.chmod = None
    _os.chown = None
    _os.chroot = None
    _os.lchflags = None
    _os.lchmod = None
    _os.lchown = None
    _os.getcwd = None
    _os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    import sys as _sys

    _sys.modules["ipdb"] = None
    _sys.modules["joblib"] = None
    _sys.modules["resource"] = None
    _sys.modules["psutil"] = None
    _sys.modules["tkinter"] = None

    for mod in ["subprocess", "ctypes"]:
        _sys.modules[mod] = None


# =============================================================================
# Helper: stdout capture
# =============================================================================


class Capturing(list):
    """Capturing implementation."""

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio
        sys.stdout = self._stdout


# =============================================================================
# Helper: string / output comparison
# =============================================================================


def truncatefn(s: str, length: int = 300) -> str:
    if len(s) <= length:
        return s
    return s[: length // 2] + "...(truncated) ..." + s[-length // 2 :]


def stripped_string_compare(s1: str, s2: str) -> bool:
    return s1.strip() == s2.strip()


def custom_compare_(output, ground_truth):
    if isinstance(output, list):
        output_1 = "\n".join(output)
        if stripped_string_compare(output_1, ground_truth):
            return True
        output_2 = [o.strip() for o in output]
        output_2 = "\n".join(output_2)
        if stripped_string_compare(output_2, ground_truth):
            return True
    return False


# =============================================================================
# Helper: call method with stdin mocked
# =============================================================================


def call_method(method, inputs):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", StringIO(inputs))
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit:
            pass

    return _inner_call_method(method)


# =============================================================================
# Local Sandbox: core run_test logic (adapted from verl prime_code)
# =============================================================================


class CODE_TYPE:
    call_based = 0
    standard_input = 1


def _run_test_core(in_outs, test, debug=False, timeout=15):
    """Internal test runner (runs inside subprocess)."""
    reliability_guard()

    if in_outs:
        if in_outs.get("fn_name") is None:
            which_type = CODE_TYPE.standard_input
            method_name = None
        else:
            which_type = CODE_TYPE.call_based
            method_name = in_outs["fn_name"]
    else:
        return [-1], {"error": "No test cases provided"}

    if test is None:
        return [-1], {"error": "Test code is None"}

    results = []
    sol = STANDARD_IMPORTS

    # ---- Compile / load the user code ----
    if which_type == CODE_TYPE.call_based:
        sol += test
        signal.alarm(timeout)
        try:
            if _HAS_PYEXT and RuntimeModule is not None:
                tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
                tmp = tmp_sol if "class Solution" not in test else tmp_sol.Solution()
            else:
                # Fallback: use exec in a fresh dict
                import sys as _sys_module
                mod_dict: dict = {"sys": _sys_module}
                exec(sol, mod_dict)
                if "class Solution" in test:
                    tmp = mod_dict["Solution"]()
                else:
                    tmp = type("_Tmp", (), mod_dict)()
            signal.alarm(0)
        except Exception as e:
            signal.alarm(0)
            error_traceback = traceback.format_exc()
            return [-2], {
                "error": repr(e),
                "traceback": error_traceback,
                "error_message": "Compilation Error",
            }

    elif which_type == CODE_TYPE.standard_input:
        # Remove if __name__ == '__main__': block
        try:
            astree = ast.parse(test)
            last_block = astree.body[-1]
            if isinstance(last_block, ast.If):
                condition = last_block.test
                if ast.unparse(condition).strip() == "__name__ == '__main__'":
                    test = ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)
        except Exception:
            pass

        tmp_test = test.split("\n")
        new_test = []
        for x in tmp_test:
            if (not x.startswith("from ")) and (not x.startswith("import ")):
                new_test.append("\t" + x + "\n")
            else:
                new_test.append(x + "\n")
        tmp_test = new_test

        new_test_str = ""
        started = False
        for i in tmp_test:
            if i.startswith("\t") and not started:
                new_test_str += "stdin = sys.stdin\nstdout = sys.stdout\n"
                new_test_str += "def code():\n"
                new_test_str += i
                started = True
            elif started and ((i.startswith("from ")) or (i.startswith("import "))):
                new_test_str += "\t" + i
            else:
                new_test_str += i
        tmp_test = new_test_str

        sol += tmp_test
        method_name = "code"
        signal.alarm(timeout)
        try:
            if _HAS_PYEXT and RuntimeModule is not None:
                tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
                tmp = tmp_sol
            else:
                import sys as _sys_module
                mod_dict = {"sys": _sys_module}
                exec(sol, mod_dict)
                tmp = type("_Tmp", (), mod_dict)()
            signal.alarm(0)
        except Exception as e:
            signal.alarm(0)
            error_traceback = traceback.format_exc()
            return [-2], {
                "error": repr(e),
                "traceback": error_traceback,
                "error_message": "Compilation Error",
            }

    # ---- Get method ----
    try:
        # In exec fallback, functions in mod_dict become bound methods
        # when accessed via getattr on the instance. Direct dict lookup
        # retrieves the raw function without auto self-injection.
        _mod_dict = mod_dict if "mod_dict" in dir() else None
        if _mod_dict is not None and method_name in _mod_dict:
            method = _mod_dict[method_name]
        else:
            method = getattr(tmp, method_name)
    except Exception:
        signal.alarm(0)
        error_info = sys.exc_info()
        return [-2], {
            "error": repr(error_info),
            "error_message": "Unable to extract code",
        }

    # ---- Run test cases ----
    for index, inputs in enumerate(in_outs["inputs"]):
        raw_inputs = inputs
        raw_outputs = in_outs["outputs"][index]

        if which_type == CODE_TYPE.call_based:
            try:
                inputs_parsed = [json.loads(line) for line in inputs.split("\n")]
                expected = json.loads(in_outs["outputs"][index])
            except Exception:
                return [-1], {"error": "Failed to parse JSON inputs for call-based test"}

            signal.alarm(timeout)
            faulthandler.enable()
            try:
                output = method(*inputs_parsed)
                if isinstance(output, tuple):
                    output = list(output)

                tmp_result = output == expected
                if isinstance(expected, list) and expected:
                    tmp_result = tmp_result or (output == expected[0])

                if tmp_result is True:
                    results.append(True)
                else:
                    return results + [False], {
                        "output": truncatefn(str(output), 200),
                        "expected": truncatefn(str(raw_outputs), 200),
                        "inputs": truncatefn(str(raw_inputs), 200),
                        "error_message": "Wrong Answer",
                    }
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                error_traceback = traceback.format_exc()
                faulthandler.disable()
                return results + [-1], {
                    "error": repr(e),
                    "traceback": error_traceback,
                }
            faulthandler.disable()

        elif which_type == CODE_TYPE.standard_input:
            faulthandler.enable()
            passed = False
            if isinstance(inputs, list):
                inputs = "\n".join(inputs)
            if isinstance(in_outs["outputs"][index], list):
                expected_output = "\n".join(in_outs["outputs"][index])
            else:
                expected_output = in_outs["outputs"][index]

            signal.alarm(timeout)
            with Capturing() as output_capture:
                try:
                    call_method(method, inputs)
                    signal.alarm(0)
                    passed = True
                except Exception as e:
                    signal.alarm(0)
                    error_traceback = traceback.format_exc()
                    return results + [-1], {
                        "error": repr(e),
                        "traceback": error_traceback,
                    }

            raw_true_output = output_capture[0] if output_capture else ""
            output_lines = raw_true_output.splitlines()

            if not passed:
                continue

            if custom_compare_(output_lines, expected_output):
                results.append(True)
                continue

            # Various comparison strategies
            tmp_result = False
            try:
                tmp_result = output_lines == [expected_output]
                if isinstance(expected_output, list):
                    tmp_result = tmp_result or (output_lines == expected_output)
                    if output_lines and isinstance(output_lines[0], str):
                        tmp_result = tmp_result or (
                            [e.strip() for e in output_lines] == expected_output
                        )
            except Exception:
                pass

            if tmp_result:
                results.append(True)
                continue

            # Split by lines and strip
            if isinstance(expected_output, list):
                expected_lines = [x.strip() for x in expected_output if x.strip()]
            else:
                expected_lines = [x.strip() for x in expected_output.split("\n") if x.strip()]

            output_lines_stripped = [x.strip() for x in output_lines if x.strip()]

            try:
                tmp_result = output_lines_stripped == expected_lines
            except Exception:
                pass

            if tmp_result:
                results.append(True)
                continue

            # Float comparison
            try:
                output_float = [float(e) for e in output_lines_stripped]
                gt_float = [float(e) for e in expected_lines]
                if len(output_float) == len(gt_float) and np.allclose(output_float, gt_float):
                    results.append(True)
                    continue
            except Exception:
                pass

            # Set comparison (word-level)
            try:
                output_set = set(" ".join(output_lines_stripped).split())
                expected_set = set(" ".join(expected_lines).split())
                if output_set == expected_set:
                    results.append(True)
                    continue
            except Exception:
                pass

            results.append(False)
            return results, {
                "output": truncatefn(raw_true_output, 200),
                "expected": truncatefn(str(raw_outputs), 200),
                "inputs": truncatefn(str(raw_inputs), 200),
                "error_message": "Wrong Answer",
            }

    return results, {}


def _temp_run(sample, generation, debug, result_list, metadata_list, timeout):
    """Entry point for multiprocessing.Process."""
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            res, metadata = _run_test_core(in_outs=sample, test=generation, debug=debug, timeout=timeout)
            result_list.append(res)
            metadata_list.append(metadata)
        except Exception:
            traceback.print_exc(10)
            result_list.append([-1 for _ in range(len(sample.get("inputs", [])))])
            metadata_list.append({})
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def local_check_correctness(in_outs: Optional[dict], generation: str, timeout: int = 10, debug: bool = False):
    """Check code correctness using local sandbox with multiprocess isolation."""
    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()

    p = multiprocessing.Process(
        target=_temp_run,
        args=(in_outs, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()

    if not result:
        num_inputs = len(in_outs.get("inputs", [])) if in_outs else 0
        return [-1] * max(num_inputs, 1), {"error": "global timeout"}

    return result[0], metadata_list


# =============================================================================
# Remote SandboxFusion
# =============================================================================


def _call_sandbox_api(
    sandbox_fusion_url: str,
    code: str,
    stdin: Optional[str],
    compile_timeout: int,
    run_timeout: int,
    memory_limit_mb: int,
    language: str = "python",
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Call remote sandbox API with retry logic."""
    if not _HAS_REQUESTS:
        return None, "requests library not installed"

    request_id = str(uuid.uuid4())
    log_prefix = f"[Request ID: {request_id}] "

    if language not in SUPPORTED_LANGUAGES:
        return None, f"Unsupported language: {language}"

    payload = json.dumps({
        "compile_timeout": compile_timeout,
        "run_timeout": run_timeout,
        "code": code,
        "stdin": stdin,
        "memory_limit_MB": memory_limit_mb,
        "language": language,
        "files": {},
        "fetch_files": [],
    })
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    request_timeout = compile_timeout + run_timeout + API_TIMEOUT

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                sandbox_fusion_url,
                headers=headers,
                data=payload,
                timeout=request_timeout,
            )
            if response.status_code == 504:
                last_error = f"Gateway Timeout (504) on attempt {attempt + 1}"
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    time.sleep(delay)
                continue
            response.raise_for_status()
            return response.json(), None
        except requests.exceptions.RequestException as e:
            last_error = f"API Request Error: {e}"
            break
        except json.JSONDecodeError as e:
            last_error = f"JSON Decode Error: {e}"
            break
        except Exception as e:
            last_error = f"Unexpected Error: {e}"
            break

    return None, last_error or "API Call Failed after retries"


def _remote_process_single_case(
    case_index: int,
    stdin_data: Any,
    expected_output: Any,
    sandbox_fusion_url: str,
    generation: str,
    timeout: int,
    memory_limit_mb: int,
    language: str,
    fn_name: Optional[str] = None,
) -> tuple[int | bool, dict[str, Any]]:
    """Process a single test case via remote sandbox."""
    current_generation_code = generation

    if fn_name and language == "python":
        wrapper_code = f'''
import traceback
{STANDARD_IMPORTS}
# === User's Original Code START ===
{generation}
# === User's Original Code END ===
_SANDBOX_FN_NAME = "{fn_name}"

def _execute_user_function():
    _raw_input_str = sys.stdin.read()
    _args = []
    if _raw_input_str.strip():
        try:
            _args = [json.loads(line) for line in _raw_input_str.split("\\n")]
        except json.JSONDecodeError as _je:
            sys.stderr.write(f"WrapperError: Invalid JSON input: {{_je}}\\n")
            return None, True
    try:
        _target_callable = None
        if _SANDBOX_FN_NAME in globals():
            _target_callable = globals()[_SANDBOX_FN_NAME]
        elif "Solution" in globals():
            _Solution_class = globals()["Solution"]
            _solution_instance = _Solution_class()
            _target_callable = getattr(_solution_instance, _SANDBOX_FN_NAME)
        if not _target_callable:
            sys.stderr.write(f"WrapperError: Function '{{_SANDBOX_FN_NAME}}' not found.\\n")
            return None, True
        _fn_result = _target_callable(*_args)
        return _fn_result, False
    except Exception:
        sys.stderr.write(f"Error during execution of '{{_SANDBOX_FN_NAME}}':\\n{{traceback.format_exc()}}\\n")
        return None, True

if __name__ == "__main__":
    _result, _error_occurred = _execute_user_function()
    if not _error_occurred:
        if isinstance(_result, (dict, list, tuple)) or _result is None or isinstance(_result, bool):
            print(json.dumps(_result))
        else:
            print(str(_result))
'''
        current_generation_code = wrapper_code

    stdin = None if stdin_data is None else str(stdin_data)
    api_response, error_msg = _call_sandbox_api(
        sandbox_fusion_url=sandbox_fusion_url,
        code=current_generation_code,
        stdin=stdin,
        compile_timeout=timeout,
        run_timeout=timeout,
        memory_limit_mb=memory_limit_mb,
        language=language,
    )

    metadata: dict[str, Any] = {
        "case_index": case_index,
        "input": stdin,
        "expected_output": str(expected_output) if expected_output else None,
        "api_request_error": error_msg,
        "status": "unknown",
        "stdout": None,
        "stderr": None,
        "run_status": None,
    }
    result_status: int | bool = -1

    if error_msg:
        metadata["status"] = "api_error"
        result_status = -1
    elif api_response:
        metadata["api_response"] = api_response
        api_status = api_response.get("status")
        compile_result = api_response.get("compile_result")
        run_result = api_response.get("run_result")

        if compile_result:
            metadata["compile_status"] = compile_result.get("status")
            metadata["compile_stderr"] = compile_result.get("stderr")
        if run_result:
            metadata["run_status"] = run_result.get("status")
            metadata["stdout"] = run_result.get("stdout")
            metadata["stderr"] = run_result.get("stderr")
            metadata["exit_code"] = run_result.get("return_code")

        if api_status == "SandboxError":
            metadata["status"] = "sandbox_error"
            result_status = -1
        elif api_status == "Failed":
            is_compile_error = compile_result and (
                metadata.get("compile_status") in ["Error", "TimeLimitExceeded"]
                or (metadata.get("compile_status") == "Finished" and compile_result.get("return_code") != 0)
            )
            if is_compile_error:
                metadata["status"] = "compile_error"
                result_status = -4
            elif run_result:
                is_runtime_error = (
                    metadata.get("run_status") == "TimeLimitExceeded"
                    or metadata.get("run_status") == "Error"
                    or (metadata.get("run_status") == "Finished" and run_result.get("return_code") != 0)
                )
                if is_runtime_error:
                    if metadata.get("run_status") == "TimeLimitExceeded":
                        metadata["status"] = "timeout"
                        result_status = -3
                    else:
                        metadata["status"] = "runtime_error"
                        result_status = -2
                else:
                    metadata["status"] = "unknown_failure"
                    result_status = -1
            else:
                metadata["status"] = "unknown_failure_state"
                result_status = -1
        elif api_status == "Success":
            if run_result and metadata.get("run_status") == "Finished":
                actual_output = metadata["stdout"] if metadata["stdout"] is not None else ""
                if expected_output is None or str(actual_output).rstrip("\n") == str(expected_output).rstrip("\n"):
                    result_status = True
                    metadata["status"] = "success"
                else:
                    result_status = False
                    metadata["status"] = "wrong_answer"
            else:
                metadata["status"] = "unexpected_success_state"
                result_status = -1
        else:
            metadata["status"] = f"unknown_api_status_{api_status}"
            result_status = -1

    return result_status, metadata


def remote_check_correctness(
    sandbox_fusion_url: str,
    in_outs: Optional[dict],
    generation: str,
    timeout: int = DEFAULT_TIMEOUT,
    memory_limit_mb: int = 1024,
    language: str = "python",
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Check correctness via remote sandbox_fusion."""
    if not in_outs or "inputs" not in in_outs or "outputs" not in in_outs:
        return [-1], [{"error": "Invalid input/output data"}]

    inputs = in_outs["inputs"]
    expected_outputs = in_outs["outputs"]
    fn_name = in_outs.get("fn_name")
    num_cases = len(inputs)

    if num_cases == 0:
        return [], []

    results = [None] * num_cases
    metadata_list = [None] * num_cases

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_cases, 32)) as executor:
        future_to_index = {
            executor.submit(
                _remote_process_single_case,
                i,
                stdin_data,
                expected_outputs[i],
                sandbox_fusion_url,
                generation,
                timeout,
                memory_limit_mb,
                language,
                fn_name,
            ): i
            for i, stdin_data in enumerate(inputs)
        }

        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result_status, metadata = future.result()
                results[index] = result_status
                metadata_list[index] = metadata
            except Exception as exc:
                logger.error(f"Test case {index} exception: {exc}")
                results[index] = -1
                metadata_list[index] = {
                    "case_index": index,
                    "error": f"Internal execution error: {exc}",
                    "status": "internal_error",
                }

    # Compile error propagation
    first_compile_error = next((i for i, r in enumerate(results) if r == -4), -1)
    if first_compile_error != -1:
        for i in range(first_compile_error + 1, num_cases):
            if results[i] != -4:
                results[i] = -4
                if metadata_list[i] is None:
                    metadata_list[i] = {"case_index": i, "status": "compile_error_skipped"}
                else:
                    metadata_list[i]["status"] = "compile_error_skipped"

    return results, metadata_list


# =============================================================================
# Unified CodeExecutor class
# =============================================================================


class CodeExecutor:
    """Code executor implementation."""

    def __init__(
        self,
        mode: str = "local",
        sandbox_fusion_url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB,
        language: str = "python",
    ):
        if mode not in ("local", "remote"):
            raise ValueError(f"mode must be 'local' or 'remote', got {mode}")
        if mode == "remote" and not sandbox_fusion_url:
            raise ValueError("sandbox_fusion_url is required when mode='remote'")

        self.mode = mode
        self.sandbox_fusion_url = sandbox_fusion_url
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb
        self.language = language
        self._semaphore = asyncio.Semaphore(32)  # Concurrency limit for remote mode

    async def execute(
        self,
        code: str,
        test_cases: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        """Execute."""
        if self.mode == "remote":
            async with self._semaphore:
                loop = asyncio.get_running_loop()
                results, metadata_list = await loop.run_in_executor(
                    None,
                    remote_check_correctness,
                    self.sandbox_fusion_url,
                    test_cases,
                    code,
                    self.timeout,
                    self.memory_limit_mb,
                    self.language,
                )
        else:
            loop = asyncio.get_running_loop()
            results, metadata_list = await loop.run_in_executor(
                None,
                local_check_correctness,
                test_cases,
                code,
                self.timeout,
                False,
            )

        # Calculate pass rate
        if not results:
            return 0.0, {"error": "No test cases executed", "metadata_list": metadata_list}

        valid_results = [r for r in results if r is True or r is False]
        if not valid_results:
            # All errors, return 0 with first error metadata
            first_error = next((m for m in metadata_list if m.get("error") or m.get("status") != "success"), {})
            return 0.0, {
                "error": "All test cases failed with errors",
                "results": results,
                "first_error": first_error,
                "metadata_list": metadata_list,
            }

        passed = sum(1 for r in valid_results if r is True)
        pass_rate = passed / len(valid_results)

        # Collect per-case metadata
        case_details = []
        for i, (res, meta) in enumerate(zip(results, metadata_list)):
            case_details.append({
                "case_index": i,
                "result": res if isinstance(res, bool) else str(res),
                "status": meta.get("status", "unknown"),
            })

        metadata = {
            "pass_rate": pass_rate,
            "passed": passed,
            "total": len(valid_results),
            "results": results,
            "case_details": case_details,
            "metadata_list": metadata_list,
        }

        return pass_rate, metadata

    @staticmethod
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
