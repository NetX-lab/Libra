"""Support code for Cmlfq tool state."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class PayloadSizeClass(Enum):
    """Payload size class implementation."""
    SizeSmall = "small"
    SizeMedium = "medium"
    SizeLarge = "large"


class ExecutionStatus(Enum):
    """Execution status implementation."""
    ToolSuccess = "success"
    ToolFailure = "failure"
    ToolTimeout = "timeout"


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolReturnState:
    """Tool return state implementation."""
    tool_type: str
    payload_size_class: PayloadSizeClass
    execution_status: ExecutionStatus
    raw_payload_size: int = 0

    def to_key(self) -> str:
        """To key."""
        return f"{self.tool_type}:{self.payload_size_class.value}:{self.execution_status.value}"

    @classmethod
    def from_key(cls, key: str) -> "ToolReturnState":
        """From key."""
        parts = key.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid ToolReturnState key: {key}")
        return cls(
            tool_type=parts[0],
            payload_size_class=PayloadSizeClass(parts[1]),
            execution_status=ExecutionStatus(parts[2]),
        )


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class ToolStateExtractor(ABC):
    """Tool state extractor implementation."""

    @abstractmethod
    def extract(self, tool_result: Any) -> ToolReturnState:
        """Extract."""
        ...

    @abstractmethod
    def can_handle(self, tool_result: Any) -> bool:
        """Can handle."""
        ...


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class DefaultToolStateExtractor(ToolStateExtractor):
    """Default tool state extractor implementation."""


    SMALL_THRESHOLD = 500
    LARGE_THRESHOLD = 5000

    def __init__(
        self,
        small_threshold: int = SMALL_THRESHOLD,
        large_threshold: int = LARGE_THRESHOLD,
    ):
        if small_threshold < 0 or large_threshold < small_threshold:
            raise ValueError("invalid payload size thresholds")
        self.small_threshold = small_threshold
        self.large_threshold = large_threshold

    def can_handle(self, tool_result: Any) -> bool:
        return isinstance(tool_result, (str, dict))

    def extract(self, tool_result: Any) -> ToolReturnState:
        if isinstance(tool_result, str):
            return self._extract_from_string(tool_result)
        elif isinstance(tool_result, dict):
            return self._extract_from_dict(tool_result)
        else:
            return ToolReturnState(
                tool_type="unknown",
                payload_size_class=PayloadSizeClass.SizeMedium,
                execution_status=ExecutionStatus.ToolSuccess,
            )

    def _extract_from_string(self, text: str) -> ToolReturnState:
        length = len(text)
        size_class = self._classify_size(length)
        status = ExecutionStatus.ToolFailure if "error" in text.lower() else ExecutionStatus.ToolSuccess
        return ToolReturnState(
            tool_type="text",
            payload_size_class=size_class,
            execution_status=status,
            raw_payload_size=length,
        )

    def _extract_from_dict(self, d: dict) -> ToolReturnState:

        tool_type = d.get("tool_type", d.get("tool", "unknown"))
        output = d.get("output", d.get("result", d.get("content", "")))
        length = int(
            d.get(
                "payload_tokens",
                len(output) if isinstance(output, str) else len(str(output)),
            )
        )
        explicit_size = str(d.get("payload_size_class", "")).lower()
        size_aliases = {
            "small": PayloadSizeClass.SizeSmall,
            "medium": PayloadSizeClass.SizeMedium,
            "large": PayloadSizeClass.SizeLarge,
        }
        size_class = size_aliases.get(
            explicit_size,
            self._classify_size(length),
        )


        explicit_status = str(d.get("status", "")).lower()
        has_error = (
            "error" in d
            or "exception" in d
            or "traceback" in d
            or explicit_status in {"error", "failure", "failed"}
            or (isinstance(output, str) and "error" in output.lower())
        )
        if explicit_status in {"timeout", "timed_out"}:
            status = ExecutionStatus.ToolTimeout
        else:
            status = (
                ExecutionStatus.ToolFailure
                if has_error
                else ExecutionStatus.ToolSuccess
            )

        return ToolReturnState(
            tool_type=str(tool_type),
            payload_size_class=size_class,
            execution_status=status,
            raw_payload_size=length,
        )

    def _classify_size(self, length: int) -> PayloadSizeClass:
        if length <= self.small_threshold:
            return PayloadSizeClass.SizeSmall
        elif length <= self.large_threshold:
            return PayloadSizeClass.SizeMedium
        else:
            return PayloadSizeClass.SizeLarge


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

class ToolStateExtractorRegistry:
    """Tool state extractor registry implementation."""

    def __init__(self):
        self._extractors: list[ToolStateExtractor] = []

    def register(self, extractor: ToolStateExtractor):
        """Register."""
        self._extractors.append(extractor)

    def extract(self, tool_result: Any) -> ToolReturnState:
        """Extract."""
        for extractor in reversed(self._extractors):
            if extractor.can_handle(tool_result):
                return extractor.extract(tool_result)

        return DefaultToolStateExtractor().extract(tool_result)

    @staticmethod
    def create_default_registry(
        small_threshold: int = DefaultToolStateExtractor.SMALL_THRESHOLD,
        large_threshold: int = DefaultToolStateExtractor.LARGE_THRESHOLD,
    ) -> "ToolStateExtractorRegistry":
        """Create default registry."""
        registry = ToolStateExtractorRegistry()
        registry.register(
            DefaultToolStateExtractor(
                small_threshold=small_threshold,
                large_threshold=large_threshold,
            )
        )
        return registry
