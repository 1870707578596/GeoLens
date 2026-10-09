# GeoLens Phase 1 —— 置信度过滤分析（熵加权 + 阈值扫描）
# 用法（打标完成后）：
#   python scripts/15_filter_analysis.py --labels data/labels.jsonl \
#       --decisions-out data/filter_decisions.jsonl --report-out data/filter_report.json
# 逻辑：
#   - 以目录名（gt）为参照，衡量 CLIP 零样本标签的可靠度
#   - 对 max_prob 做阈值扫描：每个阈值输出保留率与保留部分的精度
#   - 对熵做分位数扫描（保留低熵），作为第二视角
#   - 推荐 max_prob 阈值 = 满足精度目标（默认 0.95）的最小阈值，兼顾保留率
#   - 产出 filter_decisions.jsonl：每图 keep/drop，供 20_build_instructions.py 使用

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def sweep_max_prob(records, target):
    rows = []
    for t10 in range(20, 100, 5):
        t = t10 / 100
        kept = [r for r in records if r["max_prob"] >= t]
        prec = (
            sum(1 for r in kept if r["top1"] == r["gt"]) / len(kept) if kept else 0.0
        )
        rows.append(
            {"threshold": round(t, 2), "kept": len(kept),
             "kept_ratio": round(len(kept) / len(records), 4), "precision": round(prec, 4)}
        )
    ok = [r for r in rows if r["precision"] >= target]
    return rows, (min(ok, key=lambda r: r["threshold"]) if ok else None)


def sweep_entropy(records):
    ents = sorted(r["entropy"] for r in records)
    rows = []
    n = len(ents)
    for q10 in range(1, 10):
        q = q10 / 10
        cut = ents[min(int(n * q), n - 1)]
        kept = [r for r in records if r["entropy"] <= cut]
        prec = sum(1 for r in kept if r["top1"] == r["gt"]) / len(kept) if kept else 0.0
        rows.append(
            {"quantile": q, "entropy_cut": round(cut, 4), "kept": len(kept),
             "precision": round(prec, 4)}
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--decisions-out", required=True, type=Path)
    ap.add_argument("--report-out", required=True, type=Path)
    ap.add_argument("--precision-target", type=float, default=0.95)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.labels, encoding="utf-8") if l.strip()]
    n = len(records)
    acc = sum(1 for r in records if r["top1"] == r["gt"]) / n

    print(f"records: {n} | CLIP zero-shot top1 acc: {acc:.4f}\n")

    mp_rows, rec = sweep_max_prob(records, args.precision_target)
    print(f"{'max_prob>=':>10} {'kept':>7} {'kept%':>7} {'precision':>9}")
    for r in mp_rows:
        mark = "  <- recommended" if rec and r["threshold"] == rec["threshold"] else ""
        print(f"{r['threshold']:>10.2f} {r['kept']:>7} {r['kept_ratio']*100:>6.1f}% {r['precision']*100:>8.2f}%{mark}")

    ent_rows = sweep_entropy(records)
    print(f"\n{'entropy<=':>10} {'kept':>7} {'precision':>9}   (quantile cut)")
    for r in ent_rows:
        print(f"{r['entropy_cut']:>10.4f} {r['kept']:>7} {r['precision']*100:>8.2f}%   (q={r['quantile']})")

    # 每类精度：最差的 10 类
    per_class = defaultdict(lambda: [0, 0])
    for r in records:
        per_class[r["gt"]][0] += 1
        per_class[r["gt"]][1] += r["top1"] == r["gt"]
    worst = sorted(per_class.items(), key=lambda kv: kv[1][1] / kv[1][0])[:10]
    print("\nworst-10 classes (acc):")
    for cls, (cnt, ok) in worst:
        print(f"  {cls:<24} {ok}/{cnt} = {ok / cnt:.3f}")

    if rec is None:
        rec = {"threshold": 0.95, "kept": -1, "kept_ratio": -1, "precision": -1}
        print(f"\n[warn] 精度目标 {args.precision_target} 在扫描范围内不可达，回退 t=0.95")

    t = rec["threshold"]
    with open(args.decisions_out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(
                {k: r[k] for k in ("path", "gt", "top1", "max_prob", "entropy")}
                | {"keep": bool(r["max_prob"] >= t)}, ensure_ascii=False) + "\n")

    report = {
        "n_records": n, "clip_zero_shot_acc": round(acc, 4),
        "precision_target": args.precision_target,
        "recommended_threshold": t, "kept_at_threshold": rec["kept"],
        "kept_ratio_at_threshold": rec.get("kept_ratio"),
        "max_prob_sweep": mp_rows, "entropy_sweep": ent_rows,
        "worst_classes": [{"class": c, "n": cnt, "acc": round(ok / cnt, 4)} for c, (cnt, ok) in worst],
    }
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ndecisions -> {args.decisions_out} | report -> {args.report_out}")


if __name__ == "__main__":
    main()
