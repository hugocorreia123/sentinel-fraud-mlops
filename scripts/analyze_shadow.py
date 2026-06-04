"""Compare champion vs challenger predictions from the live service log.

Reads data/predictions.db, computes:
  - agreement rate (champion_decision == challenger_decision)
  - decision confusion matrix
  - disagreement patterns by transaction type and probability
  - McNemar's test for statistical significance of disagreement

Usage:
    uv run python -m scripts.analyze_shadow
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl
from loguru import logger
from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "predictions.db"


def load_predictions() -> pl.DataFrame:
    conn = sqlite3.connect(str(DB_PATH))
    df = pl.read_database(
        "SELECT * FROM predictions WHERE challenger_proba IS NOT NULL",
        connection=conn,
    )
    conn.close()
    return df


def agreement_summary(df: pl.DataFrame) -> dict:
    n = len(df)
    agree = (df["champion_decision"] == df["challenger_decision"]).sum()
    rate = agree / n if n else 0.0
    return {"n_total": n, "n_agree": int(agree),
            "n_disagree": int(n - agree), "agreement_rate": float(rate)}


def decision_confusion(df: pl.DataFrame) -> pl.DataFrame:
    """4-cell table: champion {APPROVE,BLOCK} × challenger {APPROVE,BLOCK}."""
    return (
        df.group_by(["champion_decision", "challenger_decision"])
        .agg(pl.len().alias("count"))
        .sort(["champion_decision", "challenger_decision"])
    )


def disagreement_by_type(df: pl.DataFrame) -> pl.DataFrame:
    disagrees = df.filter(pl.col("champion_decision") != pl.col("challenger_decision"))
    return (
        disagrees.group_by("tx_type")
        .agg(
            pl.len().alias("n_disagree"),
            pl.col("champion_proba").mean().alias("champ_proba_mean"),
            pl.col("challenger_proba").mean().alias("chall_proba_mean"),
        )
        .sort("n_disagree", descending=True)
    )


def probability_correlation(df: pl.DataFrame) -> float:
    """Pearson correlation between the two models' raw probabilities."""
    return float(
        df.select(pl.corr("champion_proba", "challenger_proba")).item()
    )


def mcnemar_significance(df: pl.DataFrame) -> dict:
    """Sign test on disagreement pairs.

    Of the rows where the two models disagree, how often is champion the one
    saying BLOCK vs challenger? Under H0 (models equally likely to flip either
    way), this is a 50/50 binomial. We use a two-sided exact binomial test —
    equivalent to McNemar's exact test on the binary disagreement pairs.
    """
    disagrees = df.filter(pl.col("champion_decision") != pl.col("challenger_decision"))
    n_disagree = len(disagrees)
    if n_disagree == 0:
        return {"n_disagree": 0, "champion_blocks_only": 0,
                "challenger_blocks_only": 0, "p_value": 1.0}

    champ_blocks_only = (disagrees["champion_decision"] == "BLOCK").sum()
    chall_blocks_only = (disagrees["challenger_decision"] == "BLOCK").sum()

    result = binomtest(int(champ_blocks_only), n=n_disagree, p=0.5, alternative="two-sided")
    return {
        "n_disagree": n_disagree,
        "champion_blocks_only": int(champ_blocks_only),
        "challenger_blocks_only": int(chall_blocks_only),
        "p_value": float(result.pvalue),
    }


def main() -> int:
    if not DB_PATH.exists():
        logger.error(f"No predictions DB at {DB_PATH}. Run the service + traffic first.")
        return 1

    df = load_predictions()
    logger.info(f"Loaded {len(df):,} predictions with shadow scoring")

    summary = agreement_summary(df)
    confusion = decision_confusion(df)
    by_type = disagreement_by_type(df)
    corr = probability_correlation(df)
    mcnemar = mcnemar_significance(df)

    print()
    print("=" * 70)
    print("Shadow-mode analysis: champion vs challenger")
    print("=" * 70)
    print(f"  Total predictions:    {summary['n_total']:,}")
    print(f"  Decisions agree:      {summary['n_agree']:,}  "
          f"({summary['agreement_rate']*100:.3f}%)")
    print(f"  Decisions disagree:   {summary['n_disagree']:,}")
    print()
    print(f"  Probability correlation (Pearson): {corr:.4f}")
    print()
    print("  Decision confusion matrix:")
    for row in confusion.iter_rows(named=True):
        print(f"    champion={row['champion_decision']:<8} "
              f"challenger={row['challenger_decision']:<8} "
              f"count={row['count']:>6}")
    print()
    print("  Disagreement breakdown by tx_type:")
    if len(by_type) == 0:
        print("    (no disagreements found)")
    else:
        for row in by_type.iter_rows(named=True):
            print(f"    {row['tx_type']:<10}  n={row['n_disagree']:>5}  "
                  f"champ_proba_mean={row['champ_proba_mean']:.4f}  "
                  f"chall_proba_mean={row['chall_proba_mean']:.4f}")
    print()
    print("  Statistical test (binomial, two-sided):")
    print(f"    H0: when the models disagree, each is equally likely to be")
    print(f"        the one calling BLOCK.")
    print(f"    Of {mcnemar['n_disagree']} disagreements: "
          f"champion blocked {mcnemar['champion_blocks_only']} times, "
          f"challenger blocked {mcnemar['challenger_blocks_only']} times.")
    print(f"    p-value: {mcnemar['p_value']:.6f}  "
          f"({'significant' if mcnemar['p_value'] < 0.05 else 'not significant'} "
          f"at α=0.05)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())