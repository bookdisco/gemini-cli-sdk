"""Internal client implementation."""

import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path

from ..types import GeminiOptions, Message, ResultMessage, SystemMessage, UserMessage
from .parser import ParserStrategy
from .parser.json_parser import JSONParser
from .parser.llm_parser import LLMParser
from .transport import Transport
from .transport.subprocess_cli import SubprocessCLITransport

logger = logging.getLogger(__name__)


class InternalClient:
    """Internal client implementation."""

    def __init__(
        self,
        transport: Transport | None = None,
        parser: ParserStrategy | None = None,
    ):
        """
        Initialize the internal client.

        Args:
            transport: Transport implementation (default: SubprocessCLITransport with JSON output)
            parser: Parser strategy (default: JSONParser for native JSON output)
        """
        self.transport = transport or SubprocessCLITransport(output_format="json")
        self.parser = parser or self._create_parser()

    def _create_parser(self) -> ParserStrategy:
        """Create parser based on environment configuration."""
        parser_type = os.getenv("GEMINI_PARSER_STRATEGY", "json").lower()

        if parser_type == "json":
            # Use native JSON parser (no extra API key needed)
            return JSONParser()
        elif parser_type == "llm":
            # Fall back to LLM parser if explicitly requested
            try:
                return LLMParser()
            except Exception as e:
                logger.warning(f"Failed to create LLM parser: {e}. Using JSON parser.")
                return JSONParser()
        else:
            logger.warning(f"Unknown parser type: {parser_type}. Using JSON parser.")
            return JSONParser()

    async def process_query(
        self, prompt: str, options: GeminiOptions
    ) -> AsyncIterator[Message]:
        """
        Process a query through transport and parser.

        Args:
            prompt: The prompt to send
            options: Configuration options

        Yields:
            Messages from the conversation
        """
        try:
            # Connect transport
            await self.transport.connect()

            # Emit initial system message
            yield SystemMessage(
                subtype="init",
                data={
                    "model": options.model
                    or os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
                    "cwd": str(options.cwd) if options.cwd else str(Path.cwd()),
                    "parser": type(self.parser).__name__,
                    "sandbox": options.sandbox,
                    "yolo": options.yolo,
                },
            )

            # Emit user message
            yield UserMessage(content=prompt)

            # Execute the query
            stdout, stderr = await self.transport.execute(prompt, options)

            # Parse the output
            messages = await self.parser.parse(stdout, stderr)

            # Yield all parsed messages
            for message in messages:
                yield message

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            # Emit error message before re-raising
            yield ResultMessage(
                subtype="error_during_execution",
                duration_ms=0,
                is_error=True,
                session_id="error",
                num_turns=0,
                result=str(e),
            )
            raise
        finally:
            # Ensure transport is disconnected
            await self.transport.disconnect()
