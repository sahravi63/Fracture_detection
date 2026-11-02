"""Command line: classify one image, a folder, or evaluate on a labelled test folder.

  python predict.py xray.jpg
  python predict.py path/to/folder --csv results.csv
  python predict.py path/to/test --evaluate          # test/<fractured|not fractured>/*.jpg
  python predict.py xray.jpg --threshold 0.3         # more sensitive (fewer missed fractures)
"""
import argparse
import csv
from pathlib import Path

from classifier import DEFAULT_MODEL, IMG_EXT, FractureClassifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="image file or folder")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--threshold", type=float, default=0.5, help="P(fractured) needed to call it fractured")
    ap.add_argument("--csv", help="write results to this CSV")
    ap.add_argument("--evaluate", action="store_true",
                    help="folder has class subfolders; report accuracy, sensitivity, specificity")
    a = ap.parse_args()

    clf = FractureClassifier(a.model)
    src = Path(a.source)
    files = [src] if src.is_file() else sorted(f for f in src.rglob("*") if f.suffix.lower() in IMG_EXT)
    if not files:
        raise SystemExit(f"No images found in {src}")

    rows, tp = [], dict(tp=0, fn=0, tn=0, fp=0)
    for f in files:
        try:
            p = clf.predict_path(f, a.threshold)
        except Exception as e:
            print(f"skipped {f.name}: {e}")
            continue
        rows.append((str(f), p.label, round(p.p_fracture, 4)))
        if a.evaluate:
            truth_fx = "not" not in f.parent.name.lower()
            pred_fx = p.label == clf.names[clf.pos]
            tp["tp" if truth_fx and pred_fx else "fn" if truth_fx else "fp" if pred_fx else "tn"] += 1
        if len(files) <= 20:
            print(f"{f.name:40s} {p.label:14s} P(fractured)={p.p_fracture:.3f}")

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "prediction", "p_fractured"])
            w.writerows(rows)
        print("saved", a.csv)

    if not rows:
        raise SystemExit("No images could be read")

    if a.evaluate:
        n = sum(tp.values())
        sens = tp["tp"] / max(tp["tp"] + tp["fn"], 1)
        spec = tp["tn"] / max(tp["tn"] + tp["fp"], 1)
        print(f"\n{n} images  accuracy={(tp['tp'] + tp['tn']) / n:.3f}  "
              f"sensitivity={sens:.3f}  specificity={spec:.3f}  {tp}")
    elif len(files) > 20:
        fx = sum(r[1] == clf.names[clf.pos] for r in rows)
        print(f"{len(rows)} images: {fx} fractured, {len(rows) - fx} not fractured")


if __name__ == "__main__":
    main()
