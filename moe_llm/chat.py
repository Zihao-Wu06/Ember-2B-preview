"""Chat format for post-training: ChatML turns, a thinking block in every
assistant turn, Hermes-style tool calls. One place renders it for the SFT data and parses it for the evals.

    <|im_start|>system\\n{system}\\n\\n<tools>\\n{one JSON tool per line}\\n</tools><|im_end|>\\n
    <|im_start|>user\\n{text}<|im_end|>\\n
    <|im_start|>assistant\\n<think>\\n{reasoning}\\n</think>\\n{answer}<|im_end|>\\n
    <|im_start|>assistant\\n<think>\\n{reasoning}\\n</think>\\n<tool_call>\\n{"name": ..., "arguments": {...}}\\n</tool_call><|im_end|>\\n
    <|im_start|>tool\\n<tool_response>\\n{result}\\n</tool_response><|im_end|>\\n

<think>, </think>, <tool_call>, </tool_call>, <tool_response>, </tool_response> are the tokenizer's reserved tokens 0-5
(one token each, no vocabulary change); <tools> ... </tools> in the system turn is plain text. The user decided on
2026-11-22 that the model always thinks, so every assistant turn opens with the block and the generation prompt
already contains "<think>\\n". An assistant message without reasoning renders as an empty block.

A conversation is {"messages": [...], "tools": [...] or None}; a message is {"role", "content"} plus, for the
assistant, "think" (str or None) and "tool_calls" ([{"name", "arguments"}] or None). Message text is stripped of
literal special tokens before rendering, so data can never inject a turn boundary.
"""
from __future__ import annotations

import json
import re

IM_START, IM_END, EOS = "<|im_start|>", "<|im_end|>", "<|endoftext|>"
THINK, THINK_END = "<|reserved_0|>", "<|reserved_1|>"
TOOL_CALL, TOOL_CALL_END = "<|reserved_2|>", "<|reserved_3|>"
TOOL_RESP, TOOL_RESP_END = "<|reserved_4|>", "<|reserved_5|>"
SPECIAL_IDS = {EOS: 0, IM_START: 2, IM_END: 3, THINK: 7, THINK_END: 8, TOOL_CALL: 9, TOOL_CALL_END: 10,
               TOOL_RESP: 11, TOOL_RESP_END: 12}
# how the reserved tokens are shown to people (logs, samples, the eval output)
ALIASES = {THINK: "<think>", THINK_END: "</think>", TOOL_CALL: "<tool_call>", TOOL_CALL_END: "</tool_call>",
           TOOL_RESP: "<tool_response>", TOOL_RESP_END: "</tool_response>"}
ROLES = ("system", "user", "assistant", "tool")
_SPECIAL = re.compile(r"<\|(?:endoftext|pad|im_start|im_end|fim_prefix|fim_middle|fim_suffix|reserved_\d)\|>")


def clean(text) -> str:
    """Message text without literal special tokens (they would be encoded as the tokens themselves)."""
    text = "" if text is None else str(text)
    return _SPECIAL.sub("", text) if "<|" in text else text


def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def tools_block(tools) -> str:
    return "<tools>\n" + "\n".join(_dumps(t) for t in tools) + "\n</tools>"


def render_assistant(msg: dict) -> str:
    """The assistant turn after "<|im_start|>assistant\\n", without the closing <|im_end|>."""
    out = THINK + "\n" + clean(msg.get("think")).strip() + "\n" + THINK_END + "\n" + clean(msg.get("content")).strip()
    for call in msg.get("tool_calls") or ():
        body = _dumps({"name": call["name"], "arguments": call.get("arguments") or {}})
        out += ("" if out.endswith("\n") else "\n") + TOOL_CALL + "\n" + clean(body) + "\n" + TOOL_CALL_END
    return out


def render_parts(conv: dict, generation_prompt: bool = False) -> list:
    """The conversation as a list of (text, no_loss) parts whose texts joined are render(conv). no_loss is True for
    the body of an assistant turn marked {"no_loss": true} (its thinking, content, tool calls and closing <|im_end|>):
    a turn that is shown to the model as context but is not a target of the training loss, e.g. a failed attempt
    that a later turn corrects (data_prep/sft_tokenize.py writes the mask, moe_llm.data applies it)."""
    msgs, tools = conv["messages"], conv.get("tools")
    if not msgs:
        raise ValueError("empty conversation")
    parts = []
    system = msgs[0]["content"] if msgs[0]["role"] == "system" else None
    if system is not None or tools:
        body = clean(system).strip()
        if tools:
            body += ("\n\n" if body else "") + clean(tools_block(tools))
        parts.append((f"{IM_START}system\n{body}{IM_END}\n", False))
    prev = "system"
    for m in msgs[1:] if system is not None else msgs:
        role = m["role"]
        if role == "system" or role not in ROLES:
            raise ValueError(f"unexpected role {role!r}")
        if role == "tool" and prev not in ("assistant", "tool"):
            raise ValueError("a tool result must follow an assistant turn")
        if role == "assistant":
            parts.append((f"{IM_START}assistant\n", False))
            parts.append((f"{render_assistant(m)}{IM_END}\n", bool(m.get("no_loss"))))
        elif role == "tool":
            parts.append((f"{IM_START}tool\n{TOOL_RESP}\n{clean(m['content']).strip()}\n{TOOL_RESP_END}{IM_END}\n", False))
        else:
            parts.append((f"{IM_START}user\n{clean(m['content']).strip()}{IM_END}\n", False))
        prev = role
    if generation_prompt:
        parts.append((f"{IM_START}assistant\n{THINK}\n", False))
    return parts


def render(conv: dict, generation_prompt: bool = False) -> str:
    """The conversation as text with the special tokens written out; ends with a newline after the last <|im_end|>
    (or with the generation prompt). Raises ValueError on a malformed conversation."""
    return "".join(text for text, _ in render_parts(conv, generation_prompt))


def no_loss_spans(conv: dict) -> list:
    """[(start, end)] character spans of render(conv) that are not training targets (see render_parts)."""
    spans, pos = [], 0
    for text, skip in render_parts(conv):
        if skip:
            spans.append((pos, pos + len(text)))
        pos += len(text)
    return spans


def parse_assistant(text: str) -> dict:
    """What the model wrote after the generation prompt -> {"think", "content", "tool_calls", "closed"}. Accepts the
    special tokens or their aliases, stops at <|im_end|>; closed is False if the thinking block never ended (then
    everything is thinking and content is empty). A tool call whose JSON does not parse is kept as {"raw": ...}."""
    for tok, alias in ALIASES.items():
        text = text.replace(alias, tok)
    for stop in (IM_END, EOS):
        text = text.split(stop)[0]
    if text.startswith(THINK):
        text = text[len(THINK):]
    closed = THINK_END in text
    think, rest = text.split(THINK_END, 1) if closed else (text, "")
    calls = []
    for body in re.findall(re.escape(TOOL_CALL) + r"(.*?)(?:" + re.escape(TOOL_CALL_END) + r"|$)", rest, re.S):
        try:
            d = json.loads(body)
            calls.append({"name": d["name"], "arguments": d.get("arguments") or {}})
        except Exception:  # noqa: BLE001
            calls.append({"raw": body.strip()})
    content = rest.split(TOOL_CALL)[0]
    return {"think": think.strip(), "content": content.strip(), "tool_calls": calls, "closed": closed}


def display(text: str) -> str:
    """Rendered text with the reserved tokens replaced by their readable aliases."""
    for tok, alias in ALIASES.items():
        text = text.replace(tok, alias)
    return text


def check_tokenizer(tok) -> None:
    """The special tokens this format uses have the ids it assumes and encode to one token each."""
    for s, i in SPECIAL_IDS.items():
        assert tok.token_to_id(s) == i, (s, tok.token_to_id(s), i)
        assert tok.encode(s).ids == [i], (s, tok.encode(s).ids)


def encode(tok, conv: dict, generation_prompt: bool = False, eos: bool = False) -> list[int]:
    ids = tok.encode(render(conv, generation_prompt)).ids
    return ids + [SPECIAL_IDS[EOS]] if eos else ids
