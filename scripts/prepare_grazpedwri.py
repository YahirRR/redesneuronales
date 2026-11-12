"""Prepara GRAZPEDWRI-DX para YOLO con partición POR PACIENTE (sin fuga entre train/val/test).

Uso:
    python scripts/prepare_grazpedwri.py --src /ruta/a/archive-2 --out data/grazpedwri

Crea enlaces simbólicos a las imágenes (no duplica ~15 GB), etiquetas YOLO filtradas a las
clases elegidas y splits.csv con la etiqueta binaria por imagen para el clasificador.
Las imágenes sin cajas de las clases elegidas quedan como negativos (etiqueta vacía).
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

NAMES = ["boneanomaly", "bonelesion", "foreignbody", "fracture", "metal",
         "periostealreaction", "pronatorsign", "softtissue", "text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="data/grazpedwri")
    ap.add_argument("--keep", nargs="+", default=["fracture"], choices=NAMES,
                    help="Clases a conservar (se reindexan en este orden)")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    remap = {NAMES.index(n): i for i, n in enumerate(a.keep)}

    images = {p.stem: p for part in sorted(src.glob("images_part*")) for p in part.glob("*.png")}
    label_dir = src / "folder_structure" / "yolov5" / "labels"
    meta = {}
    with open(src / "dataset.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            meta[r["filestem"]] = r

    stems = [s for s in images if s in meta and (label_dir / f"{s}.txt").exists()]
    print(f"{len(stems)} imágenes con metadatos y etiquetas (de {len(images)})")

    by_patient = defaultdict(list)
    for s in stems:
        by_patient[meta[s]["patient_id"]].append(s)
    patients = sorted(by_patient)
    random.Random(a.seed).shuffle(patients)
    n = len(patients)
    n_test, n_val = int(n * a.test), int(n * a.val)
    assign = {}
    for i, pid in enumerate(patients):
        assign[pid] = "test" if i < n_test else "val" if i < n_test + n_val else "train"

    rows, counts = [], defaultdict(lambda: [0, 0, 0])
    for s in stems:
        split = assign[meta[s]["patient_id"]]
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        link = out / "images" / split / images[s].name
        if not link.exists():
            link.symlink_to(images[s].resolve())

        kept = []
        for line in (label_dir / f"{s}.txt").read_text().splitlines():
            parts = line.split()
            if parts and int(parts[0]) in remap:
                kept.append(" ".join([str(remap[int(parts[0])])] + parts[1:]))
        (out / "labels" / split / f"{s}.txt").write_text("\n".join(kept) + ("\n" if kept else ""))

        fractured = int(meta[s]["fracture_visible"] == "1")
        c = counts[split]
        c[0] += 1
        c[1] += fractured
        c[2] += len(kept)
        rows.append({"path": str(link), "split": split, "label": fractured,
                     "patient_id": meta[s]["patient_id"], "projection": meta[s]["projection"],
                     "age": meta[s]["age"], "gender": meta[s]["gender"],
                     "ao_classification": meta[s]["ao_classification"]})

    with open(out / "splits.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(a.keep))
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n")

    for split in ("train", "val", "test"):
        t, fr, bx = counts[split]
        pts = sum(1 for p in assign.values() if p == split)
        print(f"{split:5s}: {pts:5d} pacientes | {t:6d} imágenes | {fr:6d} con fractura | {bx:6d} cajas")
    print(f"Listo: {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
