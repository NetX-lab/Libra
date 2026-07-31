"""Support code for Search tool."""

import asyncio
import json
import logging
import os

import aiohttp

logger = logging.getLogger("SearchTool")

SERPER_KEY = os.environ.get("SERPER_KEY_ID")
SERPER_API_URL = "https://google.serper.dev/search"
DEFAULT_SEARXNG_URL = "http://127.0.0.1:18080"


class SearchTool:
    """Search tool implementation."""

    def __init__(
        self,
        api_key: str | None = None,
        max_results: int = 5,
        backend: str | None = None,
        searxng_url: str | None = None,
    ):
        self.api_key = api_key or SERPER_KEY
        self.max_results = max_results
        self.backend = (backend or os.environ.get("SEARCH_BACKEND", "serper")).lower()
        self.searxng_url = (
            searxng_url
            or os.environ.get("SEARXNG_URL", DEFAULT_SEARXNG_URL)
        ).rstrip("/")
        if self.backend not in {"serper", "searxng"}:
            raise ValueError(
                f"Unsupported search backend: {self.backend!r}. "
                "Expected 'serper' or 'searxng'."
            )
        if self.backend == "serper" and not self.api_key:
            logger.warning(
                "SERPER_KEY_ID not set. SearchTool will return error messages. "
                "Please set the environment variable or pass api_key explicitly."
            )

    @staticmethod
    def _contains_chinese(text: str) -> bool:
        """Contains chinese."""
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    @staticmethod
    def _format_results(query: str, pages: list[dict], max_results: int) -> str:
        snippets = []
        for idx, page in enumerate(pages[:max_results], start=1):
            title = page.get("title", "")
            link = page.get("url", page.get("link", ""))
            snippet = page.get("content", page.get("snippet", ""))
            date = page.get("publishedDate", page.get("date", ""))
            date_str = f"\nDate: {date}" if date else ""
            snippets.append(f"{idx}. [{title}]({link}){date_str}\n{snippet}")

        if not snippets:
            return (
                f"No results found for query: '{query}'. "
                "Try a less specific or more general query."
            )
        return (
            f"Search results for '{query}' ({len(snippets)} results):\n\n"
            + "\n\n".join(snippets)
        )

    async def _search_serper(self, query: str) -> str:
        if not self.api_key:
            return "[Search Error] SERPER_KEY_ID not configured."

        payload = {
            "q": query,
            "gl": "cn" if self._contains_chinese(query) else "us",
            "hl": "zh-cn" if self._contains_chinese(query) else "en",
        }
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}

        last_exc = None
        async with aiohttp.ClientSession() as session:
            for attempt in range(5):
                try:
                    async with session.post(
                        SERPER_API_URL,
                        json=payload,
                        headers=headers,
                    ) as resp:
                        text = await resp.text()
                        try:
                            results = json.loads(text)
                        except Exception:
                            return f"[Search] Failed to parse response for '{query}'."

                        return self._format_results(
                            query,
                            results.get("organic", []),
                            self.max_results,
                        )

                except Exception as e:
                    last_exc = e
                    wait_time = 0.5 * (2 ** attempt)
                    logger.warning(
                        f"Search attempt {attempt + 1}/5 failed for '{query}': {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    await asyncio.sleep(wait_time)
                    continue

        return (
            f"Google search timeout or error after 5 attempts ({last_exc}). "
            "Please try again later."
        )

    async def _search_searxng(self, query: str) -> str:
        params = {
            "q": query,
            "format": "json",
            "language": "zh-CN" if self._contains_chinese(query) else "en-US",
        }
        timeout = aiohttp.ClientTimeout(total=30)
        last_exc = None
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for attempt in range(3):
                try:
                    async with session.get(
                        f"{self.searxng_url}/search",
                        params=params,
                    ) as resp:
                        resp.raise_for_status()
                        results = await resp.json(content_type=None)
                        return self._format_results(
                            query,
                            results.get("results", []),
                            self.max_results,
                        )
                except Exception as exc:
                    last_exc = exc
                    if attempt < 2:
                        await asyncio.sleep(0.5 * (2 ** attempt))

        return f"SearXNG search error after 3 attempts ({last_exc})."

    async def search(self, query: str) -> str:
        """Search."""
        if self.backend == "searxng":
            return await self._search_searxng(query)
        return await self._search_serper(query)

    async def search_batch(self, queries: list[str]) -> list[str]:
        """Search batch."""
        tasks = [self.search(q) for q in queries]
        return await asyncio.gather(*tasks)
