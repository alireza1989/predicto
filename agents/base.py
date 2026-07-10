"""Base agent class: wraps the Anthropic SDK agentic loop with tool execution."""
from __future__ import annotations

import json
import logging
import traceback
from typing import Any, Callable, Optional

import anthropic

logger = logging.getLogger(__name__)


class Tool:
    """Describes a tool that an agent can call."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict,
        func: Callable,
    ):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.func = func

    def to_api_format(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def execute(self, input_data: dict) -> str:
        """Execute the tool and return result as string."""
        try:
            result = self.func(**input_data)
            if isinstance(result, str):
                return result
            return json.dumps(result, indent=2, default=str)
        except Exception as e:
            logger.error(f"Tool {self.name} failed: {e}\n{traceback.format_exc()}")
            return json.dumps({"error": str(e), "traceback": traceback.format_exc()})


class Agent:
    """Base agent that runs an agentic loop with Claude.

    Subclasses define tools and system prompt. The agent loop:
    1. Sends messages to Claude
    2. If Claude calls tools, executes them and feeds results back
    3. Repeats until Claude returns a final text response (end_turn)
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: list[Tool],
        model: str = "claude-sonnet-5",
        max_tokens: int = 4096,
        max_iterations: int = 20,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = {t.name: t for t in tools}
        self.model = model
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.client = anthropic.Anthropic()
        self.trace: list[dict] = []  # full conversation trace

    def run(self, task: str, context: Optional[dict] = None) -> dict:
        """Run the agent on a task.

        Args:
            task: The task description/instruction
            context: Optional dict with additional context (serialized into the user message)

        Returns:
            Dict with 'response' (final text), 'trace' (full conversation), 'tool_calls' (count)
        """
        logger.info(f"[{self.name}] Starting task: {task[:100]}...")

        # Build initial user message
        user_content = task
        if context:
            user_content += f"\n\n## Context\n```json\n{json.dumps(context, indent=2, default=str)}\n```"

        messages = [{"role": "user", "content": user_content}]
        tool_call_count = 0

        api_tools = [t.to_api_format() for t in self.tools.values()]

        for iteration in range(self.max_iterations):
            logger.debug(f"[{self.name}] Iteration {iteration + 1}")

            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=self.system_prompt,
                    tools=api_tools if api_tools else anthropic.NOT_GIVEN,
                    messages=messages,
                )
            except Exception as e:
                logger.error(f"[{self.name}] API call failed: {e}")
                return {
                    "response": f"Agent error: {e}",
                    "trace": messages,
                    "tool_calls": tool_call_count,
                    "error": str(e),
                }

            # Append assistant response
            messages.append({"role": "assistant", "content": response.content})

            # Log trace
            self.trace.append({
                "iteration": iteration,
                "stop_reason": response.stop_reason,
                "content": [
                    {"type": b.type, "text": getattr(b, "text", None),
                     "tool_name": getattr(b, "name", None)}
                    for b in response.content
                ],
            })

            # If done, extract final text
            if response.stop_reason == "end_turn":
                final_text = ""
                for block in response.content:
                    if block.type == "text":
                        final_text += block.text
                logger.info(f"[{self.name}] Completed in {iteration + 1} iterations, {tool_call_count} tool calls")
                return {
                    "response": final_text,
                    "trace": self.trace,
                    "tool_calls": tool_call_count,
                }

            # Process tool calls
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_call_count += 1
                    tool_name = block.name
                    tool_input = block.input

                    logger.info(f"[{self.name}] Calling tool: {tool_name}")
                    logger.debug(f"[{self.name}] Tool input: {json.dumps(tool_input, default=str)[:500]}")

                    if tool_name in self.tools:
                        result = self.tools[tool_name].execute(tool_input)
                    else:
                        result = json.dumps({"error": f"Unknown tool: {tool_name}"})

                    # Truncate very large results
                    if len(result) > 50000:
                        result = result[:50000] + "\n... [TRUNCATED]"

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            else:
                # Stopped without end_turn and without tool calls (e.g.
                # max_tokens cut the response mid-thought). An empty user
                # message would 400 — nudge the model to wrap up instead.
                logger.warning(
                    f"[{self.name}] stop_reason={response.stop_reason} with no "
                    f"tool calls — asking model to finish"
                )
                messages.append({
                    "role": "user",
                    "content": "Your previous response was cut off. Finish your remaining work and give your conclusion concisely.",
                })

        # Max iterations reached
        logger.warning(f"[{self.name}] Hit max iterations ({self.max_iterations})")
        return {
            "response": "Agent reached maximum iterations without completing.",
            "trace": self.trace,
            "tool_calls": tool_call_count,
            "warning": "max_iterations_reached",
        }
