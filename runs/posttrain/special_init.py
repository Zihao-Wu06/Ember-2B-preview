"""Starting values for the output-head rows of the chat format's new tokens.

Pre-training never saw <|im_start|>, <|im_end|> or the reserved tokens 7-12 (their text was stripped from the data), so
the S3 checkpoint holds one and the same row for all of them, with logits near -12 .. -20: after a thought the
probability of </think> is about 1e-5 and of <|im_end|> about 1e-9. The rehearsal
(38 steps) learned the format of the thought but not to close it: 97% of the GSM8K replies never wrote </think>.
EOS ends documents in the pre-training data, so its row is trained: starting the new rows from it (plus a little noise,
so that the tokens differ) lifts the probability of </think> after a thought to 0.1%-3% and leaves the loss on ordinary
tokens unchanged (0.853 -> 0.854). The embedding rows stay as they are (zeros): copying EOS's embedding into them raised
that loss to 1.08.
"""
import torch

from moe_llm import chat

NEW_IDS = [chat.SPECIAL_IDS[t] for t in (chat.IM_START, chat.IM_END, chat.THINK, chat.THINK_END, chat.TOOL_CALL,
                                         chat.TOOL_CALL_END, chat.TOOL_RESP, chat.TOOL_RESP_END)]
EOS_ID = chat.SPECIAL_IDS[chat.EOS]


def init_rows(w: torch.Tensor, ids=NEW_IDS, eos: int = EOS_ID, scale: float = 0.1, seed: int = 0) -> torch.Tensor:
    """[len(ids), d]: the EOS row plus noise with a tenth of the per-column spread of the ordinary tokens' rows."""
    ordinary = torch.ones(w.shape[0], dtype=torch.bool)
    ordinary[list(ids) + [eos]] = False
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(len(ids), w.shape[1], generator=g) * w[ordinary].float().std(0) * scale
    return w[eos].float()[None] + noise
