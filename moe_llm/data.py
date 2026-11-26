"""Mixture data loader over data/bins/<source>/*.bin (uint16 token streams, EOS-separated).

- Each source is split into fixed windows of seq_len+1 tokens (stride seq_len); each epoch of a
  source visits its windows in a seeded random permutation (no repeats within an epoch).
- Every sequence picks its source by the mixture weights, from a RNG keyed by (seed, step):
  the state (step + per-source cursor/epoch) is tiny, so resuming is exact.
- Optional data annealing: from loader step `anneal_from` on (one step = one micro-batch) sources are
  picked by `anneal_weights` instead; train.py starts it with the LR decay. The switch depends only on
  the step counter, so it stays exact across resume and the background prefetch.
- Optional middle phase: from loader step `mid_from` until the anneal, sources are picked by `mid_weights`
  (the main run's reasoning-heavy stage 2). Same mechanism; sources only a later phase uses are appended
  to `names`, so the earlier phases keep their exact picks and window orders.
- The last `val_tokens` of every source are held out; `val_batches` gives deterministic
  per-source validation batches (math / code / English val loss separately).
- Long-document sources for context extension: a mixture name "long:<source>" draws
  windows from <source>'s stream that start at the EOS before a long document (and every seq_len after it inside
  the document), from the index data_prep/long_docs.py writes to <root>/<source>/long_docs.npz. A document gives
  windows while at least half a window of it is left, so every window is at least half one long document. Same
  permutation / cursor / resume mechanism as the plain windows; validation reads the plain source, so a long
  source has no validation batches of its own. If a directory <source>.long exists next to <source> (the compact
  upload bundle of data_prep/make_bundle.py: the long documents in a stream of their own, the plain directory only a
  sample of the short data), "long:<source>" reads that stream and its index instead; otherwise the plain one.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch

LONG = "long:"
LONG_INDEX = "long_docs.npz"
LONG_DIR = ".long"  # <source>.long: the stream "long:<source>" prefers when present (data_prep/make_bundle.py)


def long_window_starts(docs: np.ndarray, train_end: int, seq_len: int) -> np.ndarray:
    """Window starts for documents [N, 2] = (first token, length) of one stream: the EOS before the document, then
    every seq_len, while at least seq_len // 2 tokens of the document remain; a window (seq_len + 1 tokens) must end
    before train_end (the held-out tail starts there)."""
    out = []
    half = seq_len // 2
    for first, n in docs.tolist():
        if first < 1:
            continue
        k = 0
        while n - k * seq_len > half:
            w = first - 1 + k * seq_len
            if w + seq_len + 1 > train_end:
                break
            out.append(w)
            k += 1
    return np.asarray(out, dtype=np.int64)


def short_rows(x: torch.Tensor, y: torch.Tensor, names: list, short_len: int):
    """A micro-batch whose rows all come from plain sources as rows of short_len tokens ([B, L] -> [B*L/short_len,
    short_len]; the same tokens, each row's targets stay its inputs shifted by one). Context extension: the plain
    mixture windows are many short documents packed together, so they train at the short length (as in the main
    run) and only the "long:" windows train at the full length. Unchanged if short_len is 0 or a row is long."""
    if not short_len or any(n.startswith(LONG) for n in names):
        return x, y
    return x.reshape(-1, short_len), y.reshape(-1, short_len)


def load_long_index(dirs: list, total: int) -> np.ndarray:
    """(first token, length) of the long documents of a stream, from data_prep/long_docs.py's index next to its first
    .bin directory; refuses an index written for a different stream (other files or sizes)."""
    path = os.path.join(dirs[0], LONG_INDEX)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run data_prep/long_docs.py for this source first")
    z = np.load(path)
    if int(z["total"]) != total:
        raise ValueError(f"{path} was built for a stream of {int(z['total'])} tokens, the stream now has {total}")
    return z["docs"].astype(np.int64)


IGNORE = -100  # target value the loss skips (moe_llm.loss ignore_index)


def _xy(rows: list, masks: list):
    """(inputs, targets) of windows of L + 1 tokens; targets under a mask become IGNORE."""
    arr = torch.from_numpy(np.stack(rows))
    x, y = arr[:, :-1].contiguous(), arr[:, 1:].contiguous()
    if any(m is not None and m.any() for m in masks):
        # with one row per batch both slices are already contiguous, so x and y are views of the same buffer: the
        # targets must be a copy before they are written to, or the IGNORE values land in the inputs as well
        y = y.clone()
        for i, m in enumerate(masks):
            if m is not None and m.any():
                y[i][torch.from_numpy(m)] = IGNORE
    return x, y


class TokenStream:
    def __init__(self, bin_dirs):
        bin_dirs = [bin_dirs] if isinstance(bin_dirs, str) else list(bin_dirs)
        self.files = [f for d in bin_dirs for f in sorted(glob.glob(os.path.join(d, "*.bin")))]
        if not self.files:
            raise FileNotFoundError(f"no .bin files in {bin_dirs}")
        self.maps = [np.memmap(f, dtype=np.uint16, mode="r") for f in self.files]
        self.sizes = np.array([len(m) for m in self.maps], dtype=np.int64)
        self.cum = np.concatenate([[0], np.cumsum(self.sizes)])
        self.total = int(self.cum[-1])
        # optional NNN.mask next to NNN.bin (data_prep/sft_tokenize.py): uint8, 1 = this token is shown to the model but
        # is not a training target (a failed attempt that a later turn corrects). All shards of a stream or none.
        mfiles = [f[:-4] + ".mask" for f in self.files]
        have = [os.path.exists(m) for m in mfiles]
        if any(have) and not all(have):
            raise FileNotFoundError(f"some shards of {bin_dirs} have a .mask file and some do not")
        self.masks = [np.memmap(m, dtype=np.uint8, mode="r") for m in mfiles] if all(have) else None
        if self.masks is not None and [len(m) for m in self.masks] != [len(m) for m in self.maps]:
            raise ValueError(f"a .mask file of {bin_dirs} does not have the length of its .bin file")

    def read(self, start: int, length: int, maps=None) -> np.ndarray:
        maps = self.maps if maps is None else maps
        out = np.empty(length, dtype=np.int64)
        pos, filled = start, 0
        while filled < length:
            fi = int(np.searchsorted(self.cum, pos, side="right") - 1)
            off = pos - int(self.cum[fi])
            take = min(length - filled, int(self.sizes[fi]) - off)
            out[filled:filled + take] = maps[fi][off:off + take]
            filled += take
            pos += take
        return out

    def read_mask(self, start: int, length: int):
        """Boolean [length]: True where the token is not a training target; None for a stream without masks."""
        return None if self.masks is None else self.read(start, length, self.masks).astype(bool)


class MixtureLoader:
    def __init__(self, bins_root, weights: dict[str, float], seq_len: int, batch_size: int,
                 seed: int = 0, val_tokens: int = 2_000_000, anneal_weights: dict[str, float] | None = None,
                 anneal_from: int | None = None, mid_weights: dict[str, float] | None = None,
                 mid_from: int | None = None):
        # bins_root: one directory, a list, or "a,b" (e.g. data/bins,data/bins_p2 for the main run)
        if (anneal_weights is None) != (anneal_from is None):
            raise ValueError("anneal_weights and anneal_from must be given together")
        if (mid_weights is None) != (mid_from is None):
            raise ValueError("mid_weights and mid_from must be given together")
        if mid_from is not None and anneal_from is not None and mid_from > anneal_from:
            raise ValueError(f"the mid mix (from loader step {mid_from}) must start before the anneal mix "
                             f"(from {anneal_from})")
        roots = bins_root.split(",") if isinstance(bins_root, str) else list(bins_root)
        # A source's position in `names` seeds its window permutations, and the picks index into it. So the
        # stable sources come first and the sources that only a later phase uses are appended (mid, then
        # anneal); a source a later mix drops keeps its place. The earlier phases thus keep their exact batches.
        self.names = [n for n, w in weights.items() if w > 0]
        for later in (mid_weights, anneal_weights):
            self.names += [n for n, w in (later or {}).items() if w > 0 and n not in self.names]
        self.p = self._probs(weights)
        self.p_mid = self._probs(mid_weights) if mid_weights is not None else None
        self.mid_from = mid_from
        self.p_anneal = self._probs(anneal_weights) if anneal_weights is not None else None
        self.anneal_from = anneal_from
        self.L, self.B, self.seed = seq_len, batch_size, seed
        self.streams, self.starts, by_base, key_of = {}, {}, {}, {}
        for n in self.names:
            base = n[len(LONG):] if n.startswith(LONG) else n
            if n.startswith(LONG) and any(os.path.isdir(os.path.join(r, base + LONG_DIR)) for r in roots):
                base += LONG_DIR  # the long documents' own stream (compact bundle)
            if base not in by_base:
                dirs = [os.path.join(r, base) for r in roots if os.path.isdir(os.path.join(r, base))]
                by_base[base] = (dirs, TokenStream(dirs))
            key_of[n] = base
            self.streams[n] = by_base[base][1]  # otherwise a long source shares its plain source's stream
        self.val_tokens = {}
        self.n_train_windows = {}
        for n, s in self.streams.items():
            vt = min(val_tokens, s.total // 10)
            self.val_tokens[n] = vt
            if n.startswith(LONG):
                docs = load_long_index(by_base[key_of[n]][0], s.total)
                self.starts[n] = long_window_starts(docs, s.total - vt, seq_len)
                self.n_train_windows[n] = len(self.starts[n])
            else:
                self.n_train_windows[n] = (s.total - vt - 1) // seq_len
            assert self.n_train_windows[n] > 0, f"source {n} too small"
        self.state = {"step": 0, "cursor": {n: 0 for n in self.names}, "epoch": {n: 0 for n in self.names}}
        self._perm_cache = {}

    def _probs(self, weights: dict[str, float]) -> np.ndarray:
        w = np.array([max(0.0, weights.get(n, 0.0)) for n in self.names], dtype=np.float64)
        return w / w.sum()

    def phases(self) -> list[tuple[int, np.ndarray]]:
        """(first loader step, source probabilities) of each mix, in order."""
        ph = [(0, self.p)]
        if self.p_mid is not None:
            ph.append((self.mid_from, self.p_mid))
        if self.p_anneal is not None:
            ph.append((self.anneal_from, self.p_anneal))
        return ph

    def probs_at(self, step: int) -> np.ndarray:
        """Source probabilities for loader step `step` (mid mix from `mid_from`, anneal mix from `anneal_from`)."""
        if self.p_anneal is not None and step >= self.anneal_from:
            return self.p_anneal
        if self.p_mid is not None and step >= self.mid_from:
            return self.p_mid
        return self.p

    def planned_epochs(self, n_steps: int) -> dict[str, float]:
        """Expected passes over each source's training windows after `n_steps` loader steps."""
        ph = self.phases()
        draws = np.zeros(len(self.names))
        for i, (s0, p) in enumerate(ph):
            s1 = ph[i + 1][0] if i + 1 < len(ph) else n_steps
            draws = draws + self.B * max(0, min(n_steps, s1) - min(n_steps, s0)) * p
        return {n: float(d / self.n_train_windows[n]) for n, d in zip(self.names, draws)}

    def _perm(self, name: str, epoch: int) -> np.ndarray:
        ent = self._perm_cache.get(name)
        if ent is None or ent[0] != epoch:  # one cached permutation per source
            rng = np.random.default_rng([self.seed, self.names.index(name), epoch])
            ent = (epoch, rng.permutation(self.n_train_windows[name]))
            self._perm_cache[name] = ent
        return ent[1]

    def _next_window(self, name: str) -> int:
        cur, ep = self.state["cursor"][name], self.state["epoch"][name]
        if cur >= self.n_train_windows[name]:
            ep, cur = ep + 1, 0
            self.state["epoch"][name] = ep
        w = int(self._perm(name, ep)[cur])
        self.state["cursor"][name] = cur + 1
        return w

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        step = self.state["step"]
        rng = np.random.default_rng([self.seed, 1_000_003, step])
        picks = rng.choice(len(self.names), size=self.B, p=self.probs_at(step))
        self.last_names = [self.names[c] for c in picks]  # the source of each row (short_rows); not loader state
        rows, masks = [], []
        for c in picks:
            name = self.names[c]
            w = self._next_window(name)
            start = int(self.starts[name][w]) if name in self.starts else w * self.L
            rows.append(self.streams[name].read(start, self.L + 1))
            masks.append(self.streams[name].read_mask(start + 1, self.L))
        self.state["step"] += 1
        return _xy(rows, masks)

    def val_batches(self, name: str, n_batches: int):
        if name.startswith(LONG):
            return  # validated through its plain source
        s, vt = self.streams[name], self.val_tokens[name]
        base = s.total - vt
        n_win = (vt - 1) // self.L
        k = 0
        for _ in range(n_batches):
            rows, masks = [], []
            for _ in range(self.B):
                if k >= n_win:
                    break
                rows.append(s.read(base + k * self.L, self.L + 1))
                masks.append(s.read_mask(base + k * self.L + 1, self.L))
                k += 1
            if not rows:
                return
            yield _xy(rows, masks)

    def state_dict(self) -> dict:
        return {"step": self.state["step"], "cursor": dict(self.state["cursor"]), "epoch": dict(self.state["epoch"])}

    def load_state_dict(self, sd: dict) -> None:
        # Sources added after the checkpoint was written (a new anneal-only source) start at cursor 0;
        # the existing sources keep their exact state, so the stream of batches is unchanged.
        cursor = {n: 0 for n in self.names}
        cursor.update(sd["cursor"])
        epoch = {n: 0 for n in self.names}
        epoch.update(sd["epoch"])
        self.state = {"step": sd["step"], "cursor": cursor, "epoch": epoch}
