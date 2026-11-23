import asyncio
import json
from types import SimpleNamespace

import pytest

from RL_Framework.scripts.deepseek_v4_spec_rollout import speculative_config, speculative_counters


@pytest.mark.parametrize(("method", "block_size", "expected"), [
    ("mtp", None, {"method": "mtp", "num_speculative_tokens": 1}),
    ("dspark", 5, {"method": "dspark", "num_speculative_tokens": 7,
                   "draft_sample_method": "greedy"}),
])
def test_selects_compatible_v4_draft(tmp_path, method, block_size, expected):
    config = {"model_type": "deepseek_v4", "num_nextn_predict_layers": 1}
    if block_size is not None:
        config["dspark_block_size"] = block_size
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert speculative_config(tmp_path, method) == expected
    with pytest.raises(ValueError, match="requires"):
        speculative_config(tmp_path, "mtp" if method == "dspark" else "dspark")


def test_speculative_counters_aggregate_only_total_metrics():
    class Client:
        async def get(self, url):
            assert url == "http://rollout/metrics"
            return SimpleNamespace(raise_for_status=lambda: None, text=(
                '# TYPE vllm:spec_decode_num_drafts_total counter\n'
                'vllm:spec_decode_num_drafts_total{engine="0"} 4\n'
                'vllm:spec_decode_num_drafts_total{engine="1"} 6\n'
                'vllm:spec_decode_num_draft_tokens_total{engine="0"} 20\n'
                'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 12\n'
                'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 8\n'
            ))

    assert asyncio.run(speculative_counters(Client(), "http://rollout")) == {
        "drafts": 10, "draft_tokens": 20, "accepted_tokens": 12,
    }
