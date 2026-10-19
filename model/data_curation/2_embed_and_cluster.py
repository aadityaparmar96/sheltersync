"""Step 2: find duplicates and near-duplicates across ALL sources.

Two signals are combined:
  - perceptual hash (pHash): catches re-saves, resizes, recompression
  - image embedding (DINOv2 by default): catches crops, colour shifts, similar frames

Images are grouped into clusters (single-linkage over both signals). Every cluster later goes
into ONE split, which is what prevents train/test leakage. Within a cluster, images that are
the same photo are reduced to one copy (`keep` column). If copies of the same photo carry
DIFFERENT labels in different sources, both are flagged as a label conflict and dropped in step 4.

Outputs (in data/work): manifest_clustered.csv, embeddings.npy (rows align with manifest.csv)

Usage:
  python 02_embed_and_cluster.py                    # DINOv2 embeddings (use a GPU / Colab)
  python 02_embed_and_cluster.py --embedder pixels  # tiny fallback, for testing the pipeline only
"""
import argparse

import numpy as np
import pandas as pd
from PIL import Image

import config as C

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **_):
        return x

POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def compute_phashes(paths):
    import imagehash
    out = np.zeros(len(paths), dtype=np.uint64)
    for i, p in enumerate(tqdm(paths, desc="pHash")):
        with Image.open(p) as im:
            out[i] = np.uint64(int(str(imagehash.phash(im)), 16))
    return out


def embed_dinov2(paths, batch=64):
    import torch
    from transformers import AutoImageProcessor, AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(C.DINO_MODEL)
    model = AutoModel.from_pretrained(C.DINO_MODEL).to(device).eval()
    chunks = []
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), batch), desc="DINOv2"):
            imgs = []
            for p in paths[i:i + batch]:
                with Image.open(p) as im:
                    imgs.append(im.convert("RGB"))
            x = proc(images=imgs, return_tensors="pt").to(device)
            v = model(**x).pooler_output
            chunks.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)


def embed_pixels(paths):
    """Weak 16x16 grayscale embedding. For smoke tests only: use DINOv2 for real data."""
    vecs = []
    for p in tqdm(paths, desc="pixels"):
        with Image.open(p) as im:
            v = np.asarray(im.convert("L").resize((16, 16)), dtype=np.float32).ravel()
        v = (v - v.mean()) / (v.std() + 1e-6)
        vecs.append(v / (np.linalg.norm(v) + 1e-9))
    return np.stack(vecs).astype(np.float32)


def pairs_phash(H, max_dist, chunk=256):
    n, found = len(H), []
    for i in range(0, n, chunk):
        x = H[i:i + chunk, None] ^ H[None, :]
        d = POPCOUNT[x.view(np.uint8).reshape(x.shape[0], n, 8)].sum(-1)
        r, c = np.where(d <= max_dist)
        r = r + i
        m = r < c
        found.append(np.stack([r[m], c[m]], 1))
    return np.concatenate(found) if found else np.zeros((0, 2), dtype=int)


def pairs_cosine(E, thr, chunk=2048):
    n, found = len(E), []
    for i in range(0, n, chunk):
        S = E[i:i + chunk] @ E.T
        r, c = np.where(S >= thr)
        r = r + i
        m = r < c
        found.append(np.stack([r[m], c[m]], 1))
    return np.concatenate(found) if found else np.zeros((0, 2), dtype=int)


def components(n, pairs):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def hamming(a, b):
    return bin(int(a) ^ int(b)).count("1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", choices=["dinov2", "pixels"], default="dinov2")
    args = ap.parse_args()

    df = pd.read_csv(C.WORK / "manifest.csv")
    n = len(df)
    paths = df.path.tolist()

    H = compute_phashes(paths)
    E = embed_dinov2(paths) if args.embedder == "dinov2" else embed_pixels(paths)
    np.save(C.WORK / "embeddings.npy", E)

    pairs = np.concatenate([pairs_phash(H, C.CLUSTER_PHASH_HAMMING), pairs_cosine(E, C.CLUSTER_COSINE)])
    cluster = components(n, pairs)
    df["phash"] = [f"{int(h):016x}" for h in H]
    df["cluster_id"] = cluster
    df["cluster_size"] = df.groupby("cluster_id").cluster_id.transform("size")
    df["cluster_n_sources"] = df.groupby("cluster_id").source.transform("nunique")

    order = {s: i for i, s in enumerate(C.sources())}
    rank = df.source.map(order).to_numpy()
    pixels = (df.width * df.height).to_numpy()
    labels = df.mapped_label.to_numpy()
    sha = df.sha256.to_numpy()

    keep = np.ones(n, dtype=bool)
    conflict = np.zeros(n, dtype=bool)
    reason = [""] * n
    for _, idx in df.groupby("cluster_id").indices.items():
        if len(idx) == 1:
            continue
        idx = sorted(idx, key=lambda i: (rank[i], -pixels[i]))
        kept = []
        for i in idx:
            dup_of, why = None, ""
            for j in kept:
                if sha[i] == sha[j]:
                    dup_of, why = j, "exact"
                elif hamming(H[i], H[j]) <= C.DUP_PHASH_HAMMING:
                    dup_of, why = j, "phash"
                elif float(E[i] @ E[j]) >= C.DUP_COSINE:
                    dup_of, why = j, "embedding"
                if dup_of is not None:
                    break
            if dup_of is None:
                kept.append(i)
            else:
                keep[i] = False
                reason[i] = f"{why} duplicate of {paths[dup_of]}"
                if labels[i] != labels[dup_of]:
                    conflict[i] = conflict[dup_of] = True

    df["keep"] = keep
    df["dup_reason"] = reason
    df["label_conflict"] = conflict
    df.to_csv(C.WORK / "manifest_clustered.csv", index=False)

    sizes = df.cluster_size.to_numpy()
    print(f"\n{n} images -> {int(keep.sum())} after removing {int((~keep).sum())} duplicate copies")
    print(f"clusters: {df.cluster_id.nunique()}, largest: {int(sizes.max())} images")
    print(f"clusters spanning more than one source: {df[df.cluster_n_sources > 1].cluster_id.nunique()}")
    print(f"images involved in a duplicate LABEL CONFLICT: {int(conflict.sum())}")
    if sizes.max() > 50:
        print("WARNING: very large cluster. Single-linkage may be chaining unrelated images; "
              "lower CLUSTER_COSINE / CLUSTER_PHASH_HAMMING in config.py and inspect it.")
    if df.groupby("source").keep.apply(lambda s: (~s).mean()).max() > 0.3:
        print("NOTE: over 30% of a source was duplicates. Expected for remixed community datasets.")


if __name__ == "__main__":
    main()
