import asyncio

from RL_Framework.workflow.r2e_gym import R2EGymWorkflow


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, tokens, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(token) for token in tokens)

    def apply_chat_template(self, messages, **kwargs):
        del kwargs
        return "\n".join(message["content"] for message in messages)


class _Engine:
    async def generate(self, **kwargs):
        del kwargs
        return {
            "text": (
                "[ISSUE] Title: parser error\nExpected behavior: parse input.\n"
                "Actual error: parser fails in parse_input. [/ISSUE]"
            )
        }


def test_r2e_evaluation_reports_dense_reward_components(monkeypatch):
    monkeypatch.setenv("R2E_EVAL_MULTI_TURN", "0")
    workflow = R2EGymWorkflow(
        reward_fn=lambda **kwargs: 0.0,
        tokenizer=_Tokenizer(),
        max_turns=1,
        max_new_tokens=256,
    )
    dataset = [
        {
            "prompt_id": "row0",
            "prompt": "Describe the parser error.",
            "target_issue": "parser error parse input",
            "expected_output_json": '{"test_parse_input": "failed"}',
            "modified_files": ["src/parser.py"],
        }
    ]

    stats = asyncio.run(
        workflow.evaluate(
            _Engine(),
            dataset,
            max_samples=1,
            accuracy_threshold=0.5,
        )
    )

    assert stats["eval_reward_mean"] > 0.0
    assert stats["eval_index_digest"]
    assert stats["eval_lexical_f1"] > 0.0
    assert stats["eval_test_coverage"] > 0.0
    assert stats["eval_file_coverage"] > 0.0
    assert stats["eval_format_score"] > 0.0
