"""Support code for Test search r1."""

import asyncio
import json
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ------------------------------------------------------------------

# ------------------------------------------------------------------

import types
import importlib.util

if importlib.util.find_spec("torch") is None:
    torch_mock = types.ModuleType("torch")
    torch_mock.Tensor = MagicMock
    torch_mock.tensor = lambda *a, **k: MagicMock()
    torch_mock.cat = lambda tensors, dim=0: MagicMock()
    torch_mock.zeros = lambda *a, **k: MagicMock()
    torch_mock.ones = lambda *a, **k: MagicMock()
    torch_mock.float32 = "float32"
    torch_mock.__spec__ = importlib.util.spec_from_loader("torch", loader=None)
    sys.modules["torch"] = torch_mock

from env.search_tool import SearchTool
from env.search_r1_prompt import (
    SEARCH_R1_SYSTEM_PROMPT,
    SEARCH_R1_USER_PREFIX,
    TOOL_RESPONSE_TEMPLATE,
    SEARCH_R1_FORMAT_INSTRUCTION,
)
from env.search_r1_reward import (
    extract_search_r1_answer,
    extract_search_r1_think,
    search_r1_reward_fn,
    format_reward_fn,
    combined_search_r1_reward_fn,
)
from workflow.search_r1 import SearchR1Workflow


# ==================================================================

# ==================================================================

PASS_COUNT = 0
FAIL_COUNT = 0
FAILED_TESTS = []


def assert_true(condition, msg):
    global PASS_COUNT, FAIL_COUNT, FAILED_TESTS
    if condition:
        PASS_COUNT += 1
        print(f"  ✓ {msg}")
    else:
        FAIL_COUNT += 1
        FAILED_TESTS.append(msg)
        print(f"  ✗ {msg}")


def assert_eq(actual, expected, msg):
    assert_true(actual == expected, f"{msg} | expected={expected}, got={actual}")


def assert_in(substring, text, msg):
    assert_true(substring in text, f"{msg} | '{substring}' not in '{text[:80]}...'")


# ==================================================================

# ==================================================================

async def test_search_tool():
    print("\n" + "-" * 60)
    print("Test module: env/search_tool.py")
    print("-" * 60)


    print("\n[1.1] Chinese text detection")
    assert_true(
        SearchTool._contains_chinese("\u4f60\u597d\u4e16\u754c"),
        "Chinese text is detected"
    )
    assert_true(
        not SearchTool._contains_chinese("Hello World"),
        "English text is not detected as Chinese"
    )
    assert_true(
        SearchTool._contains_chinese("Hello \u4e16\u754c"),
        "Mixed Chinese-English text is detected"
    )


    print("\n[1.2] Missing API key handling")
    tool = SearchTool(api_key=None)
    result = await tool.search("test query")
    assert_in("Error", result, "A missing API key returns an error")
    assert_in("SERPER_KEY_ID", result, "The error names the environment variable")

    searxng_tool = SearchTool(
        backend="searxng",
        searxng_url="http://searxng.test:18080/",
    )
    assert_eq(searxng_tool.backend, "searxng", "The SearXNG backend is supported")
    assert_eq(
        searxng_tool.searxng_url,
        "http://searxng.test:18080",
        "The SearXNG service address is normalized",
    )


    print("\n[1.3] Successful mocked HTTP search")

    mock_response_text = json.dumps({
        "organic": [
            {"title": "Test Page 1", "link": "https://example.com/1", "snippet": "This is result 1."},
            {"title": "Test Page 2", "link": "https://example.com/2", "snippet": "This is result 2.", "date": "2024-01-01"},
        ]
    })


    def _make_mock_response(text_content):
        class Resp:
            async def text(self):
                return text_content
        class RespCtx:
            async def __aenter__(self):
                return Resp()
            async def __aexit__(self, *args):
                pass
        return RespCtx()

    class MockSession:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def post(self, *args, **kwargs):
            return _make_mock_response(mock_response_text)

    tool_with_key = SearchTool(api_key="fake_key_123")
    with patch("aiohttp.ClientSession", return_value=MockSession()):
        result = await tool_with_key.search("Python programming")

    assert_in("Search results for 'Python programming'", result, "The result contains the query heading")
    assert_in("Test Page 1", result, "The result contains the title")
    assert_in("https://example.com/1", result, "The result contains the URL")
    assert_in("This is result 1.", result, "The result contains the snippet")
    assert_in("Date: 2024-01-01", result, "The result contains the date")


    print("\n[1.4] Empty mocked HTTP results")
    empty_response = json.dumps({"searchParameters": {}})

    class MockEmptySession:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def post(self, *args, **kwargs):
            return _make_mock_response(empty_response)

    with patch("aiohttp.ClientSession", return_value=MockEmptySession()):
        result = await tool_with_key.search("xyzabc123")

    assert_in("No results found", result, "An empty result set returns a message")


    print("\n[1.5] Mocked HTTP parse failure")

    class MockBadSession:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def post(self, *args, **kwargs):
            return _make_mock_response("not json")

    with patch("aiohttp.ClientSession", return_value=MockBadSession()):
        result = await tool_with_key.search("test")

    assert_in("Failed to parse", result, "Invalid JSON returns a parse error")


    print("\n[1.6] Batch search")

    class MockBatchSession:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def post(self, *args, **kwargs):
            batch_text = json.dumps({
                "organic": [{"title": "Batch", "link": "http://b", "snippet": "batch result"}]
            })
            return _make_mock_response(batch_text)

    with patch("aiohttp.ClientSession", return_value=MockBatchSession()):
        results = await tool_with_key.search_batch(["query1", "query2"])

    assert_eq(len(results), 2, "Batch search returns one result per query")
    assert_in("batch result", results[0], "The first batch result is correct")
    assert_in("batch result", results[1], "The second batch result is correct")


# ==================================================================

# ==================================================================

def test_search_r1_reward():
    print("\n" + "-" * 60)
    print("Test module: env/search_r1_reward.py")
    print("-" * 60)


    print("\n[2.1] extract_search_r1_answer")
    assert_eq(
        extract_search_r1_answer("Some text <answer> 42 </answer> more"),
        "42",
        "Extract answer content and trim whitespace"
    )
    assert_eq(
        extract_search_r1_answer("No answer here"),
        "",
        "Return an empty value without an answer tag"
    )
    assert_eq(
        extract_search_r1_answer("<answer>Beijing</answer>"),
        "Beijing",
        "Extract an exact match"
    )
    assert_eq(
        extract_search_r1_answer("Multi\nline <answer>\n  3.14  \n</answer> end"),
        "3.14",
        "Extract multiline content and trim whitespace"
    )


    print("\n[2.2] extract_search_r1_think")
    assert_eq(
        extract_search_r1_think("<think>Let me think...</think>"),
        "Let me think...",
        "Extract think content"
    )
    assert_eq(
        extract_search_r1_think("<thinking>Deep thinking...</thinking>"),
        "Deep thinking...",
        "Extract thinking content"
    )
    assert_eq(
        extract_search_r1_think("No think tags"),
        "",
        "Return an empty value without a think tag"
    )


    print("\n[2.3] search_r1_reward_fn - correct answer")
    completion = "<think>Some reasoning</think><answer>42</answer>"
    reward = search_r1_reward_fn("prompt", completion, ground_truth="42")
    assert_eq(reward, 1.0, "Answer 42 matches ground_truth 42 and receives 1.0")

    reward = search_r1_reward_fn("prompt", completion, ground_truth=" Beijing ")
    assert_eq(reward, 0.0, "Answer 42 does not match Beijing and receives 0.0")

    completion = "<think>Some reasoning</think><answer>The Beatles</answer>"
    reward = search_r1_reward_fn(
        "prompt",
        completion,
        ground_truth=["Beatles", "The Beatles"],
    )
    assert_eq(reward, 1.0, "Multiple Search-R1 references use the official exact-match rule")


    print("\n[2.4] search_r1_reward_fn - edge cases")
    reward = search_r1_reward_fn("prompt", completion, ground_truth="")
    assert_eq(reward, 0.0, "Missing ground_truth returns 0.0")

    reward = search_r1_reward_fn("prompt", "no answer tag", ground_truth="42")
    assert_eq(reward, 0.0, "Missing answer tag returns 0.0")


    print("\n[2.5] search_r1_reward_fn - mathematical equivalence")
    comp = "<think>...</think><answer>1/2</answer>"

    reward = search_r1_reward_fn("prompt", comp, ground_truth="0.5")
    assert_true(reward > 0, "1/2 and 0.5 are treated as equivalent")


    print("\n[2.6] format_reward_fn")
    fmt = format_reward_fn("<think>...</think><answer>42</answer>")
    assert_eq(fmt, 0.15, "Think and answer tags receive the full 0.15")

    fmt = format_reward_fn("<answer>42</answer>")
    assert_eq(fmt, 0.1, "An answer tag alone receives 0.1")

    fmt = format_reward_fn("<think>...</think>")
    assert_eq(fmt, 0.05, "A think tag alone receives 0.05")

    fmt = format_reward_fn("Plain text")
    assert_eq(fmt, 0.0, "No tags receives 0.0")


    print("\n[2.7] combined_search_r1_reward_fn")
    comp = "<think>...</think><answer>42</answer>"
    reward = combined_search_r1_reward_fn("prompt", comp, ground_truth="42")
    expected = 1.0 + 0.15 * 0.15  # correctness(1.0) + format_weight(0.15) * format(0.15)
    assert_eq(reward, expected, "Combined reward equals correctness plus format reward")


# ==================================================================

# ==================================================================

def test_search_r1_prompt():
    print("\n" + "-" * 60)
    print("Test module: env/search_r1_prompt.py")
    print("-" * 60)


    print("\n[3.1] Prompt template completeness")
    assert_true(len(SEARCH_R1_SYSTEM_PROMPT) > 0, "The system prompt is not empty")
    assert_in("assistant", SEARCH_R1_SYSTEM_PROMPT, "The system prompt contains assistant")


    assert_in("<think>", SEARCH_R1_USER_PREFIX, "The user prefix contains <think>")
    assert_in("<tool_call>", SEARCH_R1_USER_PREFIX, "The user prefix contains <tool_call>")
    assert_in("<answer>", SEARCH_R1_USER_PREFIX, "The user prefix contains <answer>")
    assert_in("Question:", SEARCH_R1_USER_PREFIX, "The user prefix ends with Question")


    assert_in("tool_response", TOOL_RESPONSE_TEMPLATE, "The template contains the tool_response tag")
    assert_in("{results}", TOOL_RESPONSE_TEMPLATE, "The template contains the {results} placeholder")


    assert_in("<think>", SEARCH_R1_FORMAT_INSTRUCTION, "The format instructions mention the think tag")
    assert_in("<answer>", SEARCH_R1_FORMAT_INSTRUCTION, "The format instructions mention the answer tag")


    try:
        formatted = TOOL_RESPONSE_TEMPLATE.format(results="test results")
        assert_in("test results", formatted, "Template formatting succeeds")
    except Exception as e:
        assert_true(False, f"Template formatting failed: {e}")


# ==================================================================

# ==================================================================

async def test_search_r1_workflow():
    print("\n" + "-" * 60)
    print("Test module: workflow/search_r1.py")
    print("-" * 60)


    print("\n[4.1] _extract_tool_call")
    assert_eq(
        SearchR1Workflow._extract_tool_call("Some text <tool_call> Python tutorial </tool_call> end"),
        "Python tutorial",
        "Extract a tool_call query"
    )
    assert_eq(
        SearchR1Workflow._extract_tool_call("No tool call here"),
        None,
        "Return None without a tool_call"
    )
    assert_eq(
        SearchR1Workflow._extract_tool_call("Empty <tool_call>   </tool_call> query"),
        None,
        "Return None for an empty query"
    )
    assert_eq(
        SearchR1Workflow._extract_tool_call("Multi\n<tool_call>line\nquery</tool_call>"),
        "line\nquery",
        "Extract a multiline query"
    )


    print("\n[4.2] _has_answer_tag")
    wf = SearchR1Workflow(reward_fn=lambda **kwargs: 0.0, tokenizer=None)
    assert_true(wf._has_answer_tag("<answer>42</answer>"), "Contains an answer tag")
    assert_true(not wf._has_answer_tag("No answer"), "Does not contain an answer tag")
    assert_true(not wf._has_answer_tag("<answer>"), "An opening tag alone does not count")


    print("\n[4.3] _build_initial_prompt")


    class MockTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):

            return "\n".join([m.get("content", "") for m in messages])
        def encode(self, text, add_special_tokens=False):
            return list(text.encode("utf-8"))
        def decode(self, token_ids, skip_special_tokens=True):
            if isinstance(token_ids, list):
                return "".join(chr(b) for b in token_ids if 0 <= b <= 255)
            return ""

    tokenizer = MockTokenizer()

    wf = SearchR1Workflow(
        reward_fn=lambda **kwargs: 1.0,
        tokenizer=tokenizer,
    )
    prompt = wf._build_initial_prompt("What is 2+2?")
    assert_in("What is 2+2?", prompt, "The prompt contains the question")
    assert_in("<think>", prompt, "The prompt describes the think tag")
    assert_in("<tool_call>", prompt, "The prompt describes tool_call")
    assert_in("<answer>", prompt, "The prompt describes answer")


    print("\n[4.4] run_episode - without a tool call")

    mock_engine = AsyncMock()
    mock_engine.generate = AsyncMock(return_value={
        "text": "<think>2+2=4</think><answer>4</answer>",
        "logprobs": [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8],
    })

    wf = SearchR1Workflow(
        reward_fn=lambda prompt, completion, **kwargs: 1.0 if "4" in completion else 0.0,
        tokenizer=tokenizer,
        search_tool=SearchTool(api_key="fake"),
        max_turns=3,
    )

    trajectory = await wf.run_episode(
        engine=mock_engine,
        data={"question": "What is 2+2?", "ground_truth": "4"},
        version=0,
    )

    assert_true("input_ids" in trajectory, "The trajectory contains input_ids")
    assert_true("loss_mask" in trajectory, "The trajectory contains loss_mask")
    assert_true("rewards" in trajectory, "The trajectory contains rewards")
    assert_eq(trajectory["n_turns"], 1, "No tool call executes one turn")
    assert_eq(trajectory["n_searches"], 0, "No tool call performs no searches")
    assert_eq(trajectory["rewards"].item(), 1.0, "A correct answer receives reward 1.0")


    loss_mask = trajectory["loss_mask"][0].tolist()
    prompt_len = trajectory["input_len"]
    assert_true(prompt_len > 0, "Prompt length is greater than zero")
    assert_true(all(m == 0 for m in loss_mask[:prompt_len]), "The prompt portion of loss_mask is zero")
    assert_true(any(m == 1 for m in loss_mask[prompt_len:]), "The generated portion has loss_mask=1")


    print("\n[4.5] run_episode - with a tool call")



    call_count = 0
    async def mock_generate_with_tool(prompt, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "text": "<think>I need to search</think><tool_call>capital of France</tool_call>",
                "logprobs": [-0.1] * 15,
            }
        else:
            return {
                "text": "<think>Now I know</think><answer>Paris</answer>",
                "logprobs": [-0.2] * 12,
            }

    mock_engine2 = MagicMock()
    mock_engine2.generate = AsyncMock(side_effect=mock_generate_with_tool)

    # mock search tool
    mock_search = AsyncMock(return_value="Paris is the capital of France.")

    wf2 = SearchR1Workflow(
        reward_fn=lambda prompt, completion, **kwargs: 1.0 if "Paris" in completion else 0.0,
        tokenizer=tokenizer,
        search_tool=MagicMock(search=mock_search),
        max_turns=3,
    )

    trajectory = await wf2.run_episode(
        engine=mock_engine2,
        data={"question": "What is the capital of France?", "ground_truth": "Paris"},
        version=1,
    )

    assert_eq(trajectory["n_turns"], 2, "The episode executes two turns (search and answer)")
    assert_eq(trajectory["n_searches"], 1, "The episode performs one search")
    assert_eq(trajectory["rewards"].item(), 1.0, "A correct answer receives reward 1.0")
    assert_eq(trajectory["versions"].item(), 1, "The version is propagated correctly")


    mock_search.assert_called_once()
    call_args = mock_search.call_args[0][0]
    assert_eq(call_args, "capital of France", "The search query is extracted correctly")


    print("\n[4.6] run_episode - reaching max_turns")

    async def mock_generate_always_tool(prompt, **kwargs):
        return {
            "text": "<tool_call>search again</tool_call>",
            "logprobs": [-0.1] * 5,
        }

    mock_engine3 = MagicMock()
    mock_engine3.generate = AsyncMock(side_effect=mock_generate_always_tool)

    wf3 = SearchR1Workflow(
        reward_fn=lambda **kwargs: 0.0,
        tokenizer=tokenizer,
        search_tool=MagicMock(search=AsyncMock(return_value="some result")),
        max_turns=2,
    )

    trajectory = await wf3.run_episode(
        engine=mock_engine3,
        data={"question": "Test?", "ground_truth": "X"},
    )

    assert_eq(trajectory["n_turns"], 2, "The episode reaches max_turns")
    assert_eq(trajectory["n_searches"], 1, "The last turn does not search, for one search total")

    # ---- 4.7 run_batch ----
    print("\n[4.7] run_batch")

    mock_engine4 = AsyncMock()
    mock_engine4.generate = AsyncMock(return_value={
        "text": "<answer>OK</answer>",
        "logprobs": [-0.1] * 5,
    })

    wf4 = SearchR1Workflow(
        reward_fn=lambda **kwargs: 0.5,
        tokenizer=tokenizer,
        max_turns=1,
    )

    batch = [
        {"question": "Q1?", "ground_truth": "A1"},
        {"question": "Q2?", "ground_truth": "A2"},
    ]
    results = await wf4.run_batch(engine=mock_engine4, batch_data=batch)

    assert_eq(len(results), 2, "The batch returns two results")
    assert_eq(results[0]["rewards"].item(), 0.5, "The first result reward is correct")
    assert_eq(results[1]["rewards"].item(), 0.5, "The second result reward is correct")


# ==================================================================

# ==================================================================

async def run_all_tests():
    print("=" * 60)
    print("SearchR1 module unit tests")
    print("=" * 60)

    await test_search_tool()
    test_search_r1_reward()
    test_search_r1_prompt()
    await test_search_r1_workflow()


    print("\n" + "=" * 60)
    print("Test summary")
    print("=" * 60)
    print(f"passed: {PASS_COUNT}")
    print(f"failed: {FAIL_COUNT}")

    if FAILED_TESTS:
        print("\nFailed tests:")
        for msg in FAILED_TESTS:
            print(f"  - {msg}")
        print("\n" + "=" * 60)
        print("Some tests failed; inspect the output above.")
        print("=" * 60)
        sys.exit(1)
    else:
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
