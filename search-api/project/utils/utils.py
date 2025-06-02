from typing import List, Union
from project.models import ChatMessage


async def format_message_for_logging(
    chat_history: List[Union[ChatMessage, dict]]
) -> str:
    role_prefix = {"user": "USER: ", "assistant": "ASSISTANT: "}

    lines = ["\n"]
    for message in chat_history:
        if isinstance(message, ChatMessage):
            message = message.dict()

        role = message.get("role")
        content = message.get("content", "")

        prefix = role_prefix.get(role)
        if prefix:
            lines.append(f"{prefix}{content}\n\n")

    return "".join(lines)
