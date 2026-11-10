"""Summarise results.json files into markdown tables.

    python -m decipher.report runs/a runs/b ... > RESULTS_tables.md
"""
import json
import sys
from pathlib import Path


def main(dirs):
    for d in dirs:
        r = json.loads((Path(d) / "results.json").read_text())
        print(f"\n### {d}\n\n**Segmentation**\n\n| method | boundary F | token F | lexicon F |\n|---|---|---|---|")
        for m, v in r["segmentation"].items():
            print(f"| {m} | {v['boundary_f']:.3f} | {v['token_f']:.3f} | {v['lexicon_f']:.3f} |")
        segs = list(dict.fromkeys(k.split("/")[0] for k in r["decipherment"]))
        meths = list(dict.fromkeys(k.split("/")[1] for k in r["decipherment"]))
        print("\n**Decipherment token accuracy (rows: segmentation)**\n")
        print("| seg | " + " | ".join(meths) + " |\n|---|" + "---|" * len(meths))
        for s in segs:
            print(f"| {s} | " + " | ".join(
                f"{r['decipherment'][f'{s}/{m}']['token_acc']:.3f}" if f"{s}/{m}" in r["decipherment"] else "-"
                for m in meths) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
