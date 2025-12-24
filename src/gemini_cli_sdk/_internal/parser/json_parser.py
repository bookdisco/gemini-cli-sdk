"""JSON parser for Gemini CLI native JSON output."""

import json
import logging
from datetime import datetime

from ..._errors import CLIJSONDecodeError, ParsingError
from ...types import (
    AssistantMessage,
    ContentBlock,
    Message,
    ResultMessage,
    SystemMessage,
    TextBlock,
    UserMessage,
)
from . import ParserStrategy

logger = logging.getLogger(__name__)


class JSONParser(ParserStrategy):
    """
    Parser for native JSON output from Gemini CLI.

    Supports both single JSON output (-o json) and streaming JSON (-o stream-json).
    """

    async def parse(self, raw_output: str, stderr: str = "") -> list[Message]:
        """
        Parse JSON output from Gemini CLI.

        Args:
            raw_output: The stdout from Gemini CLI (JSON format)
            stderr: The stderr from Gemini CLI

        Returns:
            List of parsed messages
        """
        messages: list[Message] = []

        if not raw_output or not raw_output.strip():
            return messages

        # Clean output - remove non-JSON lines (like "YOLO mode is enabled...")
        cleaned_output = self._clean_output(raw_output)

        if not cleaned_output:
            return messages

        try:
            # Try to detect format: single JSON or stream-json (newline-delimited)
            if cleaned_output.strip().startswith("{") and "\n{" not in cleaned_output:
                # Single JSON object (from -o json)
                messages = self._parse_single_json(cleaned_output)
            else:
                # Stream JSON (newline-delimited from -o stream-json)
                messages = self._parse_stream_json(cleaned_output)

        except Exception as e:
            logger.error(f"JSON parsing failed: {e}")
            raise ParsingError(
                "Failed to parse Gemini CLI JSON output",
                raw_output=raw_output,
                original_error=e,
            ) from e

        return messages

    def _clean_output(self, output: str) -> str:
        """Remove non-JSON lines from output."""
        lines = output.strip().split("\n")
        json_lines = []

        for line in lines:
            stripped = line.strip()
            # Keep only lines that look like JSON
            if stripped.startswith("{") or stripped.startswith("["):
                json_lines.append(line)
            elif json_lines:
                # If we've started collecting JSON, keep continuation lines
                # (for pretty-printed JSON)
                json_lines.append(line)

        return "\n".join(json_lines)

    def _parse_single_json(self, json_str: str) -> list[Message]:
        """Parse single JSON object from -o json output."""
        messages: list[Message] = []

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise CLIJSONDecodeError(json_str, e)

        # Extract response text
        response_text = data.get("response", "")
        if response_text:
            messages.append(
                AssistantMessage(content=[TextBlock(text=response_text)])
            )

        # Extract stats for ResultMessage
        stats = data.get("stats", {})
        models_stats = stats.get("models", {})

        # Calculate totals from all models
        total_duration_ms = 0
        total_tokens = 0
        total_cost = 0.0

        for model_name, model_data in models_stats.items():
            api_stats = model_data.get("api", {})
            token_stats = model_data.get("tokens", {})

            total_duration_ms += api_stats.get("totalLatencyMs", 0)
            total_tokens += token_stats.get("total", 0)

        # Tool stats
        tool_stats = stats.get("tools", {})
        tool_calls = tool_stats.get("totalCalls", 0)

        messages.append(
            ResultMessage(
                subtype="success",
                duration_ms=total_duration_ms,
                is_error=False,
                session_id=self._generate_session_id(),
                num_turns=1,
                total_cost_usd=total_cost if total_cost > 0 else None,
                usage={
                    "total_tokens": total_tokens,
                    "tool_calls": tool_calls,
                    "models": models_stats,
                },
                result=response_text[:100] if response_text else None,
            )
        )

        return messages

    def _parse_stream_json(self, json_str: str) -> list[Message]:
        """Parse stream-json (newline-delimited JSON) output."""
        messages: list[Message] = []
        session_id = None

        for line in json_str.strip().split("\n"):
            line = line.strip()
            if not line or not line.startswith("{"):
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON line: {line[:100]}...")
                continue

            msg_type = data.get("type")

            if msg_type == "init":
                session_id = data.get("session_id", self._generate_session_id())
                messages.append(
                    SystemMessage(
                        subtype="init",
                        data={
                            "session_id": session_id,
                            "model": data.get("model"),
                            "timestamp": data.get("timestamp"),
                        },
                    )
                )

            elif msg_type == "message":
                role = data.get("role")
                content = data.get("content", "")

                if role == "user":
                    messages.append(UserMessage(content=content))
                elif role == "assistant":
                    # Check if this is a delta (partial) message
                    is_delta = data.get("delta", False)
                    if content:
                        messages.append(
                            AssistantMessage(content=[TextBlock(text=content)])
                        )

            elif msg_type == "result":
                stats = data.get("stats", {})
                messages.append(
                    ResultMessage(
                        subtype=data.get("status", "success"),
                        duration_ms=stats.get("duration_ms", 0),
                        is_error=data.get("status") == "error",
                        session_id=session_id or self._generate_session_id(),
                        num_turns=1,
                        total_cost_usd=None,  # Gemini doesn't provide cost in stream
                        usage={
                            "total_tokens": stats.get("total_tokens", 0),
                            "input_tokens": stats.get("input_tokens", 0),
                            "output_tokens": stats.get("output_tokens", 0),
                            "tool_calls": stats.get("tool_calls", 0),
                        },
                        result=None,
                    )
                )

            elif msg_type == "error":
                messages.append(
                    ResultMessage(
                        subtype="error",
                        duration_ms=0,
                        is_error=True,
                        session_id=session_id or self._generate_session_id(),
                        num_turns=0,
                        result=data.get("message", "Unknown error"),
                    )
                )

        return messages

    def _generate_session_id(self) -> str:
        """Generate a simple session ID."""
        return f"gemini-{int(datetime.now().timestamp() * 1000)}"
