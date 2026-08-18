"""Build the Artificial Stupidity corpus.

    python data/collect.py --target-mb 120
    python data/collect.py --target-mb 20 --only twitch,youtube
    python data/collect.py --list

Each source gets a fixed share of the byte budget, so the corpus mix is a
decision rather than an accident of which scraper happened to be fastest.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.sources import Sink                                          # noqa: E402
from data.sources import hf, lyrics, reddit_live, synth, twitch, youtube  # noqa: E402

RAW = ROOT / "data" / "raw"

# name -> (share of total budget, runner)
# Shares matter more than they look. Chat mode prompts the model with
# "A: ...\nB:", so ONLY the sources that emit that format can teach it to hold
# a conversation. The first version of this corpus had exactly one such source
# — the synthetic generator — and the model answered every question by
# imitating it. reddit_pairs is here to outweigh it with real humans.
MIX = {
    "twitch":       (0.20, lambda s, a: twitch.collect(
                               s, days_per_channel=a.twitch_days)),
    "youtube":      (0.20, lambda s, a: youtube.collect(
                               s, videos_per_target=a.yt_videos)),
    "reddit_pairs": (0.18, lambda s, a: hf.collect(s, "reddit_pairs")),
    "reddit":       (0.12, lambda s, a: hf.collect(s, "reddit")),
    # skips itself unless REDDIT_CLIENT_ID/SECRET are set, so keep its share
    # small enough that a skip doesn't dent the corpus
    "reddit_live":  (0.08, lambda s, a: reddit_live.collect(
                               s, pages_per_sub=a.reddit_pages)),
    "twitch_hf":    (0.08, lambda s, a: hf.collect(s, "twitch_hf")),
    "lyrics":       (0.10, lambda s, a: lyrics.collect(s)),
    "synth":        (0.04, lambda s, a: synth.collect(s)),
}


def main():
    ap = argparse.ArgumentParser(description="collect the corpus")
    ap.add_argument("--target-mb", type=float, default=100,
                    help="total corpus size to aim for, in MB")
    ap.add_argument("--only", help="comma-separated subset of sources")
    ap.add_argument("--twitch-days", type=int, default=6,
                    help="days of chat logs per Twitch channel")
    ap.add_argument("--yt-videos", type=int, default=25,
                    help="videos per YouTube channel/search")
    ap.add_argument("--reddit-pages", type=int, default=12,
                    help="PullPush pages per subreddit (100 comments each)")
    ap.add_argument("--list", action="store_true", help="show sources and exit")
    args = ap.parse_args()

    if args.list:
        print("\n  source        share   output")
        for name, (share, _) in MIX.items():
            print(f"  {name:<12}  {share:>4.0%}   data/raw/{name}.txt")
        print()
        return

    selected = [s.strip() for s in args.only.split(",")] if args.only else list(MIX)
    unknown = [s for s in selected if s not in MIX]
    if unknown:
        raise SystemExit(f"unknown source(s): {', '.join(unknown)}. "
                         f"available: {', '.join(MIX)}")

    # renormalize shares across whatever subset was requested
    total_share = sum(MIX[s][0] for s in selected)
    total_bytes = args.target_mb * 1e6

    print(f"\n{'=' * 62}")
    print(f"  COLLECTING  —  target {args.target_mb:.0f} MB across "
          f"{len(selected)} source(s)")
    print(f"{'=' * 62}")

    RAW.mkdir(parents=True, exist_ok=True)
    results = []
    t0 = time.time()

    for name in selected:
        share, runner = MIX[name]
        budget = int(total_bytes * share / total_share)
        print(f"\n  [{name}]  budget {budget / 1e6:.1f} MB")
        with Sink(RAW / f"{name}.txt", budget, label=name) as sink:
            try:
                runner(sink, args)
            except KeyboardInterrupt:
                print("      interrupted — keeping what we got")
            except Exception as e:
                print(f"      [!] {type(e).__name__}: {str(e)[:150]}")
            results.append(sink)
        print(sink.progress())

    got = sum(s.written for s in results)
    print(f"\n{'=' * 62}")
    print(f"  {'source':<14}{'size':>10}{'lines':>12}{'share':>9}")
    print(f"  {'-' * 45}")
    for s in results:
        print(f"  {s.label:<14}{s.written / 1e6:>9.2f}M{s.lines:>12,}"
              f"{100 * s.written / max(1, got):>8.1f}%")
    print(f"  {'-' * 45}")
    print(f"  {'TOTAL':<14}{got / 1e6:>9.2f}M"
          f"{sum(s.lines for s in results):>12,}")
    print(f"\n  elapsed {time.time() - t0:.0f}s")
    print(f"\n  next:  python data/prepare.py\n")


if __name__ == "__main__":
    main()
