"""
Drift simulation: compares normal IMDB data (reference) vs drifted data (current).
Runs standalone - no running API required.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drift.features import extract_features_batch
from drift.psi import compute_psi, interpret_psi

VEC_PATH = "models/tfidf_vectorizer.pkl"
DATA_PATH = "data/raw/IMDB Dataset.csv"
OUTPUT_PATH = "reports/drift_report.png"
REF_STATS_PATH = "drift/reference_stats.json"

FEATURE_NAMES = ["text_length", "word_count", "unknown_ratio"]


def load_reference(vocab: set, n: int = 1000) -> dict:
    df = pd.read_csv(DATA_PATH).sample(n, random_state=42)
    feats = extract_features_batch(df["review"].tolist(), vocab)
    return {f: [x[f] for x in feats] for f in FEATURE_NAMES}


def load_drifted(vocab: set, n: int = 300) -> dict:
    # Simulate covariate shift: modern slang + tech jargon not in IMDB vocab
    drifted_texts = [
        "lol omg bruh lowkey sus ngl vibe check fr fr no cap bussin",
        "glitch lag bug server timeout crash error 404 kernel panic segfault",
        "hashtag trending viral tiktok reel shorts influencer collab merch drop",
        "yo this slaps hard deadass periodt slay queen no printer just fax",
        "api endpoint microservice kubernetes docker container orchestration devops",
    ] * (n // 5)

    feats = extract_features_batch(drifted_texts[:n], vocab)
    return {f: [x[f] for x in feats] for f in FEATURE_NAMES}


def main():
    os.makedirs("reports", exist_ok=True)

    vectorizer = joblib.load(VEC_PATH)
    vocab = set(vectorizer.vocabulary_.keys())

    print("Loading reference distribution (IMDB training data)...")
    reference = load_reference(vocab)

    print("Generating drifted data distribution...")
    drifted = load_drifted(vocab)

    # Save reference stats for future comparisons
    ref_stats = {
        f: {"mean": round(float(np.mean(reference[f])), 4),
            "std": round(float(np.std(reference[f])), 4)}
        for f in FEATURE_NAMES
    }
    with open(REF_STATS_PATH, "w") as fp:
        json.dump(ref_stats, fp, indent=2)

    print("\n" + "=" * 55)
    print("  PSI DRIFT DETECTION REPORT")
    print("=" * 55)
    print(f"{'Feature':<25} {'PSI (normal)':>12} {'PSI (drifted)':>14} {'Status'}")
    print("-" * 55)

    results = []
    for feat in FEATURE_NAMES:
        # PSI of a held-out slice of reference vs reference (should be ~0)
        psi_normal = compute_psi(reference[feat][:800], reference[feat][800:])
        psi_drifted = compute_psi(reference[feat], drifted[feat])
        status = interpret_psi(psi_drifted)
        results.append({
            "feature": feat,
            "psi_normal": psi_normal,
            "psi_drifted": psi_drifted,
            "status": status,
        })
        print(f"{feat:<25} {psi_normal:>12.4f} {psi_drifted:>14.4f}   {status}")

    print("=" * 55)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("PSI Drift Detection: Normal vs Drifted Data", fontsize=13)

    colors_normal = ["#2ecc71"]   # green
    colors_drifted = []
    for r in results:
        p = r["psi_drifted"]
        colors_drifted.append("#e74c3c" if p >= 0.25 else ("#f39c12" if p >= 0.10 else "#2ecc71"))

    for i, (feat, res) in enumerate(zip(FEATURE_NAMES, results)):
        ax = axes[i]
        bars = ax.bar(
            ["Normal\n(PSI vs ref)", "Drifted\n(PSI vs ref)"],
            [res["psi_normal"], res["psi_drifted"]],
            color=[colors_normal[0], colors_drifted[i]],
            width=0.5,
        )
        ax.axhline(0.10, color="#f39c12", linestyle="--", linewidth=1, label="Warning 0.10")
        ax.axhline(0.25, color="#e74c3c", linestyle="--", linewidth=1, label="Alarm 0.25")
        ax.set_title(feat, fontsize=11)
        ax.set_ylabel("PSI")
        ax.legend(fontsize=7)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=100, bbox_inches="tight")
    print(f"\nPlot saved: {OUTPUT_PATH}")
    print(f"Reference stats saved: {REF_STATS_PATH}")


if __name__ == "__main__":
    main()
