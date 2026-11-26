"""Download the post-training datasets to
data/raw/sft/<name>/ with download_curl.py's resumable curl fetch; every LFS file is checked against the Hub's SHA-256.

    python3 data_prep/download_sft.py            # download and verify
    python3 data_prep/download_sft.py --list     # only print what would be fetched
    python3 data_prep/download_sft.py --only rstar_coder   # some sets; the manifest is MANIFEST_<names>.json
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import download_curl as dc  # noqa: E402

# name -> (repo, path prefixes to fetch ("" = the whole repo), licence as listed on the Hub on 2026-11-22)
SETS = {
    "openmathinstruct2_1m": ("nvidia/OpenMathInstruct-2", ("data/train_1M-", "README.md"), "cc-by-4.0"),
    "openthoughts_114k": ("open-thoughts/OpenThoughts-114k", ("data/", "README.md"), "apache-2.0"),
    "smoltalk_all": ("HuggingFaceTB/smoltalk", ("data/all/", "README.md"), "apache-2.0 (README)"),
    "glaive_fc_v2": ("glaiveai/glaive-function-calling-v2", ("",), "apache-2.0"),
    "hermes_fc_v1": ("NousResearch/hermes-function-calling-v1", ("",), "apache-2.0"),
    "toolace": ("Team-ACE/ToolACE", ("",), "apache-2.0"),
    "apps": ("codeparrot/apps", ("",), "mit"),
    "self_oss_instruct_50k": ("bigcode/self-oss-instruct-sc2-exec-filter-50k", ("data/", "README.md"), "odc-by"),
    "gsm8k": ("openai/gsm8k", ("main/", "README.md"), "mit"),
    "math500": ("HuggingFaceH4/MATH-500", ("",), "not stated (MATH is MIT)"),
    # the train split only, 24 files, 2.27 GB; the converter keeps the Gutenberg books
    "narrativeqa": ("deepmind/narrativeqa", ("data/train-", "README.md"), "apache-2.0"),
    # rStar-Coder: code problems with a reasoning and a solution checked by mutual
    # verification on generated tests; the synthetic_sft config only, 15 files, 6.51 GB
    "rstar_coder": ("microsoft/rStar-Coder", ("synthetic_sft/", "README.md"), "cc-by-4.0"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=[], choices=list(SETS), help="fetch these sets only")
    a = ap.parse_args()
    out = os.path.join(dc.ROOT, "data", "raw", "sft")
    jobs, meta = [], {}
    for name, (repo, prefixes, lic) in SETS.items():
        if a.only and name not in a.only:
            continue
        meta[repo] = dc.remote_files(repo)
        files = [f for f in meta[repo] if any(f.startswith(p) for p in prefixes) and f != ".gitattributes"]
        size = sum(meta[repo][f]["size"] for f in files)
        dc.log(f"{name}: {repo}, {len(files)} files, {size / 1e9:.2f} GB, licence {lic}")
        jobs += [(name, repo, f, os.path.join(out, name, f)) for f in files]
    total = sum(meta[r][f]["size"] for _, r, f, _ in jobs)
    dc.log(f"{len(jobs)} files, {total / 1e9:.2f} GB -> {out}")
    if a.list:
        return

    def job(repo, f, dest):
        m = meta[repo][f]
        res = dc.fetch(repo, f, dest, m["size"])
        if m["sha256"] and dc.sha256_of(dest) != m["sha256"]:
            os.remove(dest)
            raise RuntimeError("SHA-256 mismatch, file deleted")
        return res, bool(m["sha256"])

    manifest, failed = [], []
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(job, r, f, d): (n, r, f) for n, r, f, d in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            n, r, f = futs[fut]
            try:
                (path, size, dt, cached), checked = fut.result()
            except Exception as e:  # noqa: BLE001
                failed.append(f"{r}/{f}")
                dc.log(f"FAIL {r}/{f}: {e}")
                continue
            manifest.append({"set": n, "repo": r, "file": f, "path": os.path.relpath(path, dc.ROOT), "bytes": size,
                             "sha256_checked": checked, "licence": SETS[n][2]})
            dc.log(f"OK {i}/{len(jobs)} {r}/{f} {size / 1e6:.0f} MB{' (already here)' if cached else f' in {dt:.0f}s'}"
                   f"{', SHA-256 ok' if checked else ''}")
    with open(os.path.join(out, f"MANIFEST{'_' + '_'.join(a.only) if a.only else ''}.json"), "w", encoding="utf-8") as fh:
        json.dump({"files": manifest, "failed": failed, "bytes": sum(m["bytes"] for m in manifest)}, fh, indent=1)
    dc.log(f"SFT_DOWNLOAD_DONE: {len(manifest)} ok, {len(failed)} failed, "
           f"{sum(m['bytes'] for m in manifest) / 1e9:.2f} GB")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
