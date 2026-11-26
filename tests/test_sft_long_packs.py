"""data_prep/sft_long_packs.py: packing conversations into the loader's long: windows."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data_prep"))
pytest.importorskip("tokenizers")
if not os.path.exists(os.path.join(ROOT, "data", "tokenizer", "tokenizer.json")):
    pytest.skip("tokenizer not built", allow_module_level=True)
import sft_long_packs as P  # noqa: E402
from moe_llm import chat  # noqa: E402
from moe_llm.data import MixtureLoader  # noqa: E402


def test_pack():
    bins = P.pack([70, 10, 60, 30, 200, 40, 5], 100)
    assert bins == [[0, 3], [2, 5], [1, 6]]  # 70+30, 60+40, 10+5; the item of 200 is left out
    assert P.pack([], 10) == []


def test_packs_feed_the_loader(tmp_path, monkeypatch):
    conv, bins = tmp_path / "conv", tmp_path / "bins"
    conv.mkdir()
    (bins / "_medium").mkdir(parents=True)
    rows = [{"id": f"l/{i}", "messages": [{"role": "user", "content": f"doc {i} " + "word " * (40 + 25 * (i % 7))},
                                           {"role": "assistant", "think": "t", "content": f"answer {i}"}]} for i in range(60)]
    with open(conv / "long_records.jsonl", "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(P.TOK)
    med = [np.asarray(chat.encode(tok, {"messages": [{"role": "user", "content": "medium " * 150},
                                                     {"role": "assistant", "think": "m", "content": "ok"}]}, eos=True),
                      dtype=np.uint16)]
    np.savez(bins / "_medium" / "sft_x.npz", ids=np.concatenate(med), offsets=np.array([0, len(med[0])]))
    np.savez(bins / "_medium" / "sft_other.npz", ids=np.concatenate(med), offsets=np.array([0, len(med[0])]))
    # packs of chosen set-aside conversations only (--sets with no name: no conversation set)
    monkeypatch.setattr(sys, "argv", ["x", "--bins", str(bins), "--conv", str(conv), "--pack-len", "256", "--name",
                                      "only_medium", "--sets", "--medium", "sft_x", "sft_other", "missing"])
    P.main()
    om = json.load(open(bins / "only_medium" / "meta.json"))
    assert set(om["sets"]) == {"medium/sft_x", "medium/sft_other"} and om["packs"] >= 1
    monkeypatch.setattr(sys, "argv", ["x", "--bins", str(bins), "--conv", str(conv), "--pack-len", "512",
                                      "--medium", "sft_x"])
    P.main()
    meta = json.load(open(bins / "sft_long" / "meta.json"))
    z = np.load(bins / "sft_long" / "long_docs.npz")
    ids = np.fromfile(bins / "sft_long" / "000.bin", dtype=np.uint16)
    assert int(z["total"]) == len(ids) == meta["tokens"] and ids[0] == 0 and len(z["docs"]) == meta["packs"] > 3
    seen = []
    for first, n in z["docs"].tolist():
        assert 256 < n <= 512 and ids[first - 1] == 0 and ids[first + n - 1] == 0  # EOS before, ends with a whole conversation
        seen += tok.decode(ids[first:first + n].tolist(), skip_special_tokens=False).split("<|endoftext|>")[:-1]
    want = [chat.render(r) for r in rows] + [tok.decode(med[0][:-1].tolist(), skip_special_tokens=False)]
    assert len(seen) == len(set(seen)) and set(seen) <= set(want)   # whole conversations, none twice
    assert len(seen) >= len(want) - 8                               # only an underfilled last pack may be left out
    ld = MixtureLoader(str(bins), {"long:sft_long": 1.0}, seq_len=512, batch_size=2, val_tokens=600)
    x, _ = ld.next_batch()
    starts = set((z["docs"][:, 0] - 1).tolist())
    assert x.shape == (2, 512) and all(int(s) in starts for s in ld.starts["long:sft_long"])
    assert (x[:, 0] == 0).all() and (x[:, 1] == 2).all()  # a window starts at the EOS before a pack, then <|im_start|>
