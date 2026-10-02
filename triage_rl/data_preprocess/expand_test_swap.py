#!/usr/bin/env python
"""Expand a prompt-only test JSON so each stay appears twice: survival-first and
death-first.

The evaluation harness averages the two reasoning orders of the same stay. Without the
expansion the aggregator pairs consecutive rows, which are different stays, and reports
metrics over fabricated pairs without raising.

The swap below rewrites the survival / in-hospital-death wording, so it is a no-op on
P19, whose prompts are phrased around sepsis.

Output feeds TRIAGE_dataset.py as --test_data_source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re

DEATH_MARKER = "1. If the patient indeed experiences in-hospital death,"
SURVIVAL_MARKER = "1. If the patient indeed survives,"


def swap_prompt_rationale_order(prompt_text):
    if not isinstance(prompt_text, str):
        return None

    prompt_text = re.sub(
        r"1\.\s*If the patient indeed survives, which of the patient's given features might be the cause\?\n"
        r"2\.\s*If the patient indeed experiences in-hospital death, which of the patient's given features might be the cause\?",
        "1. If the patient indeed experiences in-hospital death, which of the patient's given features might be the cause?\n"
        "2. If the patient indeed survives, which of the patient's given features might be the cause?",
        prompt_text,
        count=1,
    )

    prompt_text = prompt_text.replace(
        "## Rationale for survival\n[possible justification if patient survives]\n\n"
        "## Rationale for in-hospital death\n[possible justification if patient experiences in-hospital death]\n\n",
        "## Rationale for in-hospital death\n[possible justification if patient experiences in-hospital death]\n\n"
        "## Rationale for survival\n[possible justification if patient survives]\n\n",
        1,
    )

    return prompt_text
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-json", required=True, help="prompt-only test JSON, one record per stay")
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    swap = swap_prompt_rationale_order
    src = json.load(open(args.in_json))
    print(f"[expand] {args.in_json}: {len(src)} records")

    out = []
    noop = missing_in_swapped = marker_in_canonical = 0
    for r in src:
        if list(r.keys()) != ["MOR_label", "prompt"]:
            raise SystemExit(f"unexpected record keys {list(r.keys())}; expected ['MOR_label','prompt']")
        if len(r["prompt"]) != 1 or r["prompt"][0]["role"] != "user":
            raise SystemExit("expected a single user turn per record")
        canonical = r["prompt"][0]["content"]
        if SURVIVAL_MARKER not in canonical:
            raise SystemExit("input record is not survival-first; refusing to expand")
        if DEATH_MARKER in canonical:
            marker_in_canonical += 1
        swapped = swap(canonical)
        if swapped == canonical:
            noop += 1
        if DEATH_MARKER not in (swapped or ""):
            missing_in_swapped += 1
        out.append({"MOR_label": r["MOR_label"], "prompt": [{"role": "user", "content": canonical}]})
        out.append({"MOR_label": r["MOR_label"], "prompt": [{"role": "user", "content": swapped}]})

    # The same gates the original build asserts; counts only,
    # never the offending text -- these files are credentialed patient-derived data.
    for name, n in (
        ("swap_noop", noop),
        ("death_marker_missing_in_swapped", missing_in_swapped),
        ("death_marker_in_canonical", marker_in_canonical),
    ):
        if n:
            raise SystemExit(f"[expand] GATE FAILED: {name}={n}")
    if len(out) != 2 * len(src):
        raise SystemExit(f"[expand] GATE FAILED: {len(out)} records from {len(src)} stays")

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(out, fh)
    h = hashlib.sha256(open(args.out_json, "rb").read()).hexdigest()
    print(
        f"[expand] wrote {args.out_json}: {len(out)} records "
        f"({len(src)} stays x 2), gates swap_noop=0 missing=0 canonical_marker=0"
    )
    print(f"[expand] sha256 {h}")


if __name__ == "__main__":
    main()
