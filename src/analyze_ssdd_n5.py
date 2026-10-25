# -*- coding: utf-8 -*-
"""
SSDD n5 配对统计（baseline vs dia_l2_k3，k=3 最终配置）
======================================================
读 ssdd_multiseed_n5 下各任务 val_metrics.csv，5 配对种子 [42,123,2024,7,2025]。
对 mAP@0.5 / mAP@0.5:0.95 / AP-small / AP-medium / AP-large 计算：
- mean ± std (ddof=1)
- 配对 t 检验 (scipy.stats.ttest_rel)
- Wilcoxon 符号秩检验 (scipy.stats.wilcoxon)
- 双侧符号检验 (scipy.stats.binomtest, H0: P(+) = 0.5)
- 配对效应量 Cohen's d_z = mean(d)/std(d)
- Bonferroni 阈值 α=0.05/5=0.01
输出：逐配对表、mean±std、检验结果；写 ssdd_paired_stats.csv。
"""
import csv
import math
from pathlib import Path
from scipy import stats

OUT = Path(r"D:\SAR_Det_Research\ssdd_multiseed_n5")
SEEDS = [42, 123, 2024, 7, 2025]
METRICS = ["mAP@0.5", "mAP@0.5:0.95", "AP-small", "AP-medium", "AP-large"]
VARIANTS = ["baseline", "dia_l2_k3"]
ALPHA = 0.05
ALPHA_BONF = ALPHA / len(METRICS)


def mean(x):
    return sum(x) / len(x)


def sample_std(x):
    m = mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))


def load(variant):
    grp = {}
    for s in SEEDS:
        p = OUT / f"SSDD_{variant}_seed{s}" / "val_coco_result" / "val_metrics.csv"
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        r = rows[0]
        grp[s] = {k: float(r[k]) for k in METRICS}
    return grp


def main():
    base = load("baseline")
    dia = load("dia_l2_k3")
    common = sorted(set(base) & set(dia))
    n = len(common)
    print(f"共同种子数 n={n}: {common}")
    if n < 2:
        print("配对数不足，等待训练完成。")
        return

    # 逐配对表
    print("\n" + "=" * 90)
    print("逐配对结果（dia - baseline）")
    print("=" * 90)
    header = f"{'seed':>6} | " + " | ".join(f"{k:^14}" for k in METRICS)
    print(header)
    for s in common:
        cells = [f"{base[s][k]:.4f}->{dia[s][k]:.4f}" for k in METRICS]
        print(f"{s:>6} | " + " | ".join(f"{c:^14}" for c in cells))
    print("-" * 90)
    for s in common:
        cells = [f"{(dia[s][k]-base[s][k])*100:+8.2f}pt" for k in METRICS]
        print(f"{s:>6} | " + " | ".join(f"{c:^14}" for c in cells))

    # mean ± std
    print("\n" + "=" * 90)
    print("mean ± std (n=%d)" % n)
    print("=" * 90)
    summ = {}
    for name, grp in [("baseline", base), ("dia", dia)]:
        print(f"-- {name} --")
        for k in METRICS:
            vals = [grp[s][k] for s in common]
            print(f"  {k:14} {mean(vals):.4f} ± {sample_std(vals):.4f}")
            summ.setdefault(k, {})[name] = (mean(vals), sample_std(vals))

    # 配对检验
    print("\n" + "=" * 90)
    print("配对检验（dia - baseline，差值百分点 = diff*100）")
    print("=" * 90)
    rows_out = []
    for k in METRICS:
        d = [dia[s][k] - base[s][k] for s in common]
        md = mean(d)
        sd = sample_std(d)
        tstat, pt = stats.ttest_rel([dia[s][k] for s in common],
                                    [base[s][k] for s in common])
        # Wilcoxon（有零差值时需 zero_method 处理；n 小）
        try:
            wstat, pw = stats.wilcoxon([dia[s][k] for s in common],
                                       [base[s][k] for s in common],
                                       zero_method="wilcox", alternative="two-sided",
                                       method="exact")
        except Exception as e:
            wstat, pw = (float("nan"), float("nan"))
        # 符号检验：正/负（忽略零差值）
        npos = sum(1 for v in d if v > 0)
        nneg = sum(1 for v in d if v < 0)
        nz = sum(1 for v in d if v == 0)
        nsign = npos + nneg
        if nsign > 0:
            kk = max(npos, nneg)
            psign = stats.binomtest(kk, nsign, 0.5, alternative="two-sided").pvalue
        else:
            psign = 1.0
        d_z = md / sd if sd > 0 else float("inf")
        sig_t = "SIG" if pt < ALPHA else "ns"
        sig_b = "Bonf-SIG" if pt < ALPHA_BONF else ""
        print(f"{k:14} diff={md*100:+.2f}pt  s_d={sd*100:.3f}  "
              f"t={tstat:+.2f} p_t={pt:.4f} {sig_t} | Wilcoxon p={pw:.4f} | "
              f"sign +{npos}/-{nneg}/0{nz} p={psign:.4f} | d_z={d_z:.2f} {sig_b}")
        rows_out.append({
            "metric": k, "n": n,
            "baseline_mean": round(summ[k]["baseline"][0], 4),
            "baseline_std": round(summ[k]["baseline"][1], 4),
            "dia_mean": round(summ[k]["dia"][0], 4),
            "dia_std": round(summ[k]["dia"][1], 4),
            "diff_points": round(md * 100, 3),
            "sd_diff_points": round(sd * 100, 4),
            "t": round(tstat, 3), "p_t": round(pt, 4),
            "p_wilcoxon": round(pw, 4) if not math.isnan(pw) else "NA",
            "pos": npos, "neg": nneg, "zero": nz,
            "p_sign": round(psign, 4),
            "cohen_dz": round(d_z, 3) if d_z != float("inf") else "inf",
            "sig_alpha05": sig_t,
            "sig_bonferroni_0.01": "yes" if pt < ALPHA_BONF else "no",
        })

    out_csv = OUT / "ssdd_paired_stats.csv"
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\n已保存：{out_csv}")
    print(f"Bonferroni 阈值（5 指标）：p < {ALPHA_BONF:.3f}")


if __name__ == "__main__":
    main()
