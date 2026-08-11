"""Support code for Rollout engine."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class MockRolloutEngine:
    """Small rollout engine for dependency and training-loop smoke tests."""

    def __init__(
        self,
        model_path: str = "",
        text: str | None = None,
    ):
        self.model_path = model_path
        self.text = text or os.environ.get("MOCK_ROLLOUT_TEXT", " The answer is 0.")
        self.num_instances = 1
        self.instance_urls = ["mock://local"]
        self.tp_list = [1]

    def wait_for_ready(self, timeout: float = 0.0):
        del timeout
        logger.info("Mock rollout engine is ready")

    async def close(self):
        return None

    def close_sync(self):
        return None

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
        input_tokens: int = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del prompt, max_new_tokens, temperature, top_p, n, input_tokens, kwargs
        return {
            "text": self.text,
            "tokens": [],
            "logprobs": [],
            "finish_reason": "mock",
        }


class VLLMRolloutEngine:
    """V l l m rollout engine implementation."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        model_path: str = "",
    ):
        self.host = host
        self.port = port
        self.model_path = model_path

        # URL
        self.base_url = f"http://{self.host}:{self.port}"
        self.health_url = f"{self.base_url}/health"
        self.completions_url = f"{self.base_url}/v1/completions"


        self.http_client: httpx.AsyncClient | None = None
        self._http_client_loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def wait_for_ready(self, timeout: float = 300.0):
        """Wait for ready."""
        import requests

        start_time = time.time()
        check_count = 0
        logger.info(f"Waiting for the vLLM service: {self.health_url}")

        while time.time() - start_time < timeout:
            check_count += 1
            try:
                response = requests.get(self.health_url, timeout=5)
                if response.status_code == 200:
                    logger.info(
                        f"The vLLM service is ready ({check_count} checks, "
                        f"elapsed{time.time() - start_time:.0f}s)"
                    )
                    return
            except Exception:
                if check_count % 15 == 0:
                    elapsed = time.time() - start_time
                    logger.info(
                        f"Waiting for vLLM to start... "
                        f"(waited {elapsed:.0f}s, {check_count} checks)"
                    )
            time.sleep(2.0)

        raise TimeoutError(
            f"The vLLM service did not become ready (waited{timeout}s), address: {self.base_url}"
        )

    async def _ensure_client(self):
        """Ensure client."""
        current_loop = asyncio.get_running_loop()
        if (
            self.http_client is None
            or self.http_client.is_closed
            or self._http_client_loop is not current_loop
        ):
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=float(os.environ.get("VLLM_CONNECT_TIMEOUT", "30")),
                    read=float(os.environ.get("VLLM_READ_TIMEOUT", "600")),
                    write=float(os.environ.get("VLLM_WRITE_TIMEOUT", "30")),
                    pool=float(os.environ.get("VLLM_POOL_TIMEOUT", "30")),
                )
            )
            self._http_client_loop = current_loop

    async def close(self):
        """Close."""
        if self.http_client is not None and not self.http_client.is_closed:
            await self.http_client.aclose()
            self.http_client = None
            self._http_client_loop = None

    def close_sync(self):
        """Best-effort synchronous close used during runtime reconfiguration."""
        if self.http_client is None or self.http_client.is_closed:
            self.http_client = None
            self._http_client_loop = None
            return
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.http_client.aclose())
            else:
                loop.run_until_complete(self.http_client.aclose())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self.http_client.aclose())
            finally:
                loop.close()
        finally:
            self.http_client = None
            self._http_client_loop = None

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
        input_tokens: int = 0,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Generate."""
        await self._ensure_client()

        payload = {
            "model": self.model_path,
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "logprobs": 1,
        }
        if seed is not None:
            payload["seed"] = int(seed)

        try:
            response = await self.http_client.post(
                self.completions_url,
                json=payload,
            )

            if response.status_code != 200:
                error_text = response.text
                raise RuntimeError(
                    f"Generation failed: status={response.status_code}, {error_text}"
                )

            result = response.json()
            return self._parse_response(result)

        except Exception as e:
            logger.error(
                "Generation request failed: %s: %r "
                "(prompt_chars=%d, max_new_tokens=%d, url=%s)",
                type(e).__name__,
                e,
                len(prompt or ""),
                max_new_tokens,
                self.completions_url,
            )
            raise

    async def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
    ) -> list[dict[str, Any]]:
        """Generate batch."""
        tasks = [
            self.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
            )
            for prompt in prompts
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Prompt {i} Generation failed: {result}")
                continue
            valid_results.append(result)

        return valid_results

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def _parse_response(self, response: dict) -> dict[str, Any]:
        """Parse response."""
        if not isinstance(response, dict):
            raise ValueError(f"Invalid response type: {type(response).__name__}")

        choices = response.get("choices", [])
        if not choices:
            raise ValueError("Empty response")

        choice = choices[0]
        if not isinstance(choice, dict):
            raise ValueError(f"Invalid choice type: {type(choice).__name__}")

        text = choice.get("text", "")
        logprobs_data = choice.get("logprobs", {})
        if logprobs_data is None:
            logger.warning("vLLM response omitted token logprobs; keeping generated text")
            logprobs_data = {}
        elif not isinstance(logprobs_data, dict):
            logger.warning(
                "vLLM response returned non-dict logprobs (%s); keeping generated text",
                type(logprobs_data).__name__,
            )
            logprobs_data = {}

        tokens = logprobs_data.get("tokens") or []
        token_logprobs = logprobs_data.get("token_logprobs") or []

        return {
            "text": text or "",
            "tokens": tokens,
            "logprobs": token_logprobs,
            "finish_reason": choice.get("finish_reason", "length"),
        }

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def __del__(self):
        """Del."""
        if self.http_client is not None and not self.http_client.is_closed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.http_client.aclose())
                else:
                    loop.run_until_complete(self.http_client.aclose())
            except Exception:
                pass


class MultiInstanceRolloutEngine:
    """Multi instance rollout engine implementation."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        base_port: int = 8000,
        num_instances: int = 1,
        model_path: str = "",
        endpoints: list[str] | None = None,
    ):
        self.model_path = model_path
        self.engines: list[VLLMRolloutEngine] = []

        if endpoints:

            for ep in endpoints:
                ep = ep.strip()
                if ":" in ep:
                    h, p = ep.rsplit(":", 1)
                    engine = VLLMRolloutEngine(
                        host=h, port=int(p), model_path=model_path
                    )
                else:
                    raise ValueError(f"Invalid endpoint format: '{ep}', expected 'host:port'")
                self.engines.append(engine)
            self.num_instances = len(self.engines)
            self.host = "multi-host"
            self.base_port = 0
            logger.info(
                f"MultiInstanceRolloutEngine (multi-host): {self.num_instances} instances, "
                f"endpoints: {[e.base_url for e in self.engines]}"
            )
        else:

            self.host = host
            self.base_port = base_port
            self.num_instances = max(1, num_instances)
            for i in range(self.num_instances):
                engine = VLLMRolloutEngine(
                    host=host,
                    port=base_port + i,
                    model_path=model_path,
                )
                self.engines.append(engine)
            logger.info(
                f"MultiInstanceRolloutEngine (single-host): {self.num_instances} instances, "
                f"port{base_port}-{base_port + self.num_instances - 1}"
            )


        self._counter = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def _next_engine(self) -> VLLMRolloutEngine:
        """Next engine."""
        with self._lock:
            engine = self.engines[self._counter % self.num_instances]
            self._counter += 1
        return engine

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def wait_for_ready(self, timeout: float = 300.0):
        """Wait for ready."""
        for i, engine in enumerate(self.engines):
            logger.info(f"Waiting for vLLM instance {i}/{self.num_instances} to become ready...")
            engine.wait_for_ready(timeout=timeout)
        logger.info(f"All {self.num_instances} vLLM instances are ready")

    async def close(self):
        """Close."""
        for engine in self.engines:
            await engine.close()

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
        input_tokens: int = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate."""
        engine = self._next_engine()
        return await engine.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            input_tokens=input_tokens,
            **kwargs,
        )

    async def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
    ) -> list[dict[str, Any]]:
        """Generate batch."""
        tasks = [
            self.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
            )
            for prompt in prompts
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Prompt {i} Generation failed: {result}")
                continue
            valid_results.append(result)

        return valid_results

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    @property
    def instance_urls(self) -> list[str]:
        """Instance urls."""
        return [e.base_url for e in self.engines]

    def __repr__(self):
        return (
            f"MultiInstanceRolloutEngine("
            f"instances={self.num_instances}, "
            f"ports={self.base_port}-{self.base_port + self.num_instances - 1})"
        )
