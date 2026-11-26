"""runs/posttrain/special_init.py: starting rows for the chat format's new tokens."""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "runs", "posttrain"))
sys.path.insert(0, ROOT)
import special_init as S  # noqa: E402


def test_init_rows():
    g = torch.Generator().manual_seed(1)
    w = torch.randn(64, 32, generator=g) * 2.0
    w[S.NEW_IDS] = -0.3                                     # the shared, untrained row of the new tokens
    rows = S.init_rows(w)
    assert rows.shape == (len(S.NEW_IDS), 32) and torch.equal(rows, S.init_rows(w))   # deterministic
    cos = torch.nn.functional.cosine_similarity(rows, w[S.EOS_ID][None], dim=1)
    assert bool((cos > 0.97).all())                         # close to the EOS row ...
    assert float((rows[0] - rows[1]).abs().max()) > 0       # ... but the tokens differ from each other
    assert not torch.equal(S.init_rows(w, seed=1), rows)
    assert S.NEW_IDS == [2, 3, 7, 8, 9, 10, 11, 12] and S.EOS_ID == 0
