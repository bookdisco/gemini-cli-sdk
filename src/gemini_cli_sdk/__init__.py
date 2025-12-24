"""Gemini SDK for Python - Compatible with Claude Code SDK.

This SDK provides a Python interface for the Gemini CLI, with an API
design compatible with the Claude Code SDK for easy migration.

Features:
- Native JSON output parsing (no extra API key needed)
- Session management with --resume support
- Tool permissions with --allowed-tools
- Sandbox and YOLO modes
- Async streaming interface

Example:
    ```python
    from gemini_cli_sdk import query, GeminiOptions

    async for message in query(
        prompt="Hello, Gemini!",
        options=GeminiOptions(yolo=True)
    ):
        print(message)
    ```
"""

import os
from collections.abc import AsyncIterator

from ._errors import (
    # Compatibility aliases
    ClaudeSDKError,
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    ConfigurationError,
    GeminiSDKError,
    ParsingError,
    ProcessError,
)
from ._internal.client import InternalClient
from .types import (
    # Main types
    AssistantMessage,
    # Compatibility alias
    ClaudeCodeOptions,
    CodeBlock,
    ContentBlock,
    GeminiOptions,
    Message,
    PermissionMode,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

__version__ = "0.2.0"

__all__ = [
    # Main function
    "query",
    # Types
    "PermissionMode",
    "UserMessage",
    "AssistantMessage",
    "SystemMessage",
    "ResultMessage",
    "Message",
    "GeminiOptions",
    "ClaudeCodeOptions",  # Compatibility alias
    "TextBlock",
    "CodeBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "ContentBlock",
    # Errors
    "GeminiSDKError",
    "ClaudeSDKError",  # Compatibility alias
    "CLIConnectionError",
    "CLINotFoundError",
    "ProcessError",
    "CLIJSONDecodeError",
    "ParsingError",
    "ConfigurationError",
]


async def query(
    *, prompt: str, options: GeminiOptions | None = None
) -> AsyncIterator[Message]:
    """
    Query Gemini CLI.

    Python SDK for interacting with Gemini CLI, with an API compatible
    with Claude Code SDK.

    Args:
        prompt: The prompt to send to Gemini
        options: Optional configuration (defaults to GeminiOptions() if None).
                 Set options.model to choose Gemini model (default: gemini-2.0-flash).
                 Set options.yolo=True to auto-accept all actions.
                 Set options.sandbox=True to run in sandbox mode.
                 Set options.resume="latest" to resume a previous session.
                 Set options.allowed_tools=["tool1", "tool2"] to allow specific tools.
                 Set options.cwd for working directory.

    Yields:
        Messages from the conversation

    Example:
        ```python
        # Simple usage
        async for message in query(prompt="Hello"):
            print(message)

        # With options
        async for message in query(
            prompt="Hello",
            options=GeminiOptions(
                model="gemini-2.0-flash",
                yolo=True,
                cwd="/home/user"
            )
        ):
            print(message)

        # Resume a session
        async for message in query(
            prompt="Continue our conversation",
            options=GeminiOptions(resume="latest")
        ):
            print(message)
        ```

    Note:
        This SDK uses Gemini CLI's native JSON output format for parsing,
        so no additional API keys are required beyond what Gemini CLI needs.

        Set GEMINI_PARSER_STRATEGY=llm to use LLM-based parsing instead
        (requires GEMINI_API_KEY or GOOGLE_API_KEY).
    """
    if options is None:
        options = GeminiOptions()

    # Set SDK identifier
    os.environ["GEMINI_CODE_SDK"] = "python"

    client = InternalClient()

    async for message in client.process_query(prompt=prompt, options=options):
        yield message
