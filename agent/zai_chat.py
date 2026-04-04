"""LangChain chat model wrapping Z.ai ``zai-sdk`` (GLM via ``ZaiClient``)."""

from __future__ import annotations

from typing import Any, cast

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


def _message_content_to_text(content: str | list[str | dict[Any, Any]]) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
        else:
            parts.append(str(block))
    return "".join(parts)


def _messages_to_zai(messages: list[BaseMessage]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for m in messages:
        role = m.type
        if role == "system":
            r = "system"
        elif role == "human":
            r = "user"
        elif role == "ai":
            r = "assistant"
        else:
            r = "user"
        text = _message_content_to_text(cast("str | list[str | dict[Any, Any]]", m.content))
        out.append({"role": r, "content": text})
    return out


class ChatZai(BaseChatModel):
    """Chat completions via ``ZaiClient`` (e.g. GLM-4.7-Flash)."""

    model: str = Field(default="glm-4.7-flash")
    temperature: float = Field(default=0.2)
    api_key: str = Field(repr=False)
    base_url: str | None = Field(
        default=None,
        description="Override API base (default: SDK / ZAI_BASE_URL env).",
    )

    @property
    def _llm_type(self) -> str:
        return "zai-chat"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del kwargs  # reserved for LangChain; not forwarded to Z.ai
        try:
            from zai import ZaiClient
        except ImportError as e:
            raise ImportError("Z.ai chat requires: pip install zai-sdk") from e

        client_kw: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kw["base_url"] = self.base_url
        client = ZaiClient(**client_kw)

        zai_messages = _messages_to_zai(messages)
        create_kw: dict[str, Any] = {
            "model": self.model,
            "messages": zai_messages,
            "temperature": self.temperature,
        }
        if stop:
            create_kw["stop"] = stop

        response = client.chat.completions.create(**create_kw)
        text = response.choices[0].message.content
        if text is None:
            text = ""
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=text))],
        )
