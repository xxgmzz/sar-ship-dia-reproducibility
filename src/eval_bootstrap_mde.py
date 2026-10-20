# -*- coding: utf-8 -*-
"""
阶段A：噪声地板 / 两层噪声分离 / MDE / 所需种子数 分析（零训练）
====================================================================
产出（输出根目录：D:\\SAR_Det_Research\\bootstrap_mde_analysis）
1) 每个数据集：gt_coco.json + 代表模型(seed42, baseline/dia) 的 det_*.json
2) 图像级 bootstrap（有放回重采样图像，重算 COCO AP）=> 测试集抽样噪声
3) 运行间(run-to-run)噪声地板：读多种子 runs CSV，算 mean/std
4) 配对差(dia-baseline)：mean、配对差std(s_d)、配对t、p
5) MDE 表：
   - 单次运行(独立比较) 90/95% 临界差异
   - n 种子配对检验 显著性临界 / 含80%功效的 MDE
6) 所需种子数：目标效应 d（配对，alpha=.05, power=.8）；std估计相对精度目标
用法：
   自测：  python run_bootstrap_mde_analysis.py smoke
   全量：  python run_bootstrap_mde_analysis.py
"""
import os
import io
import csv
import sys
import math
import json
import contextlib
from pathlib import Path

import numpy as np
from PIL import Image

# ====================== 配置区 ======================
PY_NOTE = "det"
IMG_SIZE = 640
CONF_THRES = 0.001
IOU_THRES = 0.6
DEVICE = 0
BOOT_B = 500                 # bootstrap 次数
BOOT_SEED = 20261006
ALPHA = 0.05
POWER = 0.80
Z95 = 1.959964               # 正态双侧 .05
Z90 = 1.644854               # 正态双侧 .10

OUT_FULL = Path(r"D:\SAR_Det_Research\bootstrap_mde_analysis")
OUT_SMOKE = Path(r"D:\SAR_Det_Research\bootstrap_mde_smoke")

# 指标：内部key -> runs CSV 列名；COCO stats 索引见 coco_stats()
METRICS = [("mAP50", "mAP@0.5"),
           ("mAP5095", "mAP@0.5:0.95"),
           ("APs", "AP-small"),
           ("APm", "AP-medium"),
           ("APl", "AP-large")]
MKEYS = [m[0] for m in METRICS]
MCOL = dict(METRICS)

# 数据集配置
# model_tmpl: root\{prefix}_{tagdir}_seed{seed}\weights\best.pt
DATASETS = [
    {
        "name": "HRSID",
        "img_dir": r"D:\SAR_Det_Research\hrsid_experiments\data\images\val",
        "lab_dir": r"D:\SAR_Det_Research\hrsid_experiments\data\labels\val",
        "runs_csv": r"D:\SAR_Det_Research\multiseed_experiments\multiseed_runs.csv",
        "model_col": "variant",
        "runs_filter": {"dataset": "HRSID"},
        "runs_tags": {"baseline": "baseline", "dia": "dia_l2_k3"},
        "model_root": r"D:\SAR_Det_Research\multiseed_experiments",
        "model_prefix": "HRSID",
        "model_tagdir": {"baseline": "baseline", "dia": "dia_l2_k3"},
        "rep_seed": 42,
    },
    {
        "name": "Cross_SSDD",
        "img_dir": r"D:\SAR_Det_Research\exp00_baseline_yolov8n_ssdd\sar_ssdd\images\val",
        "lab_dir": r"D:\SAR_Det_Research\exp00_baseline_yolov8n_ssdd\sar_ssdd\labels\val",
        "runs_csv": r"D:\SAR_Det_Research\crossdomain_hrsid_to_ssdd\crossdomain_runs.csv",
        "model_col": "group",
        "runs_filter": {},
        "runs_tags": {"baseline": "baseline", "dia": "dia"},
        # 跨域：复用 HRSID 训练的同批模型
        "model_root": r"D:\SAR_Det_Research\multiseed_experiments",
        "model_prefix": "HRSID",
        "model_tagdir": {"baseline": "baseline", "dia": "dia_l2_k3"},
        "rep_seed": 42,
    },
    {
        "name": "HRSC",
        "img_dir": r"D:\SAR_Det_Research\hrsc2016_yolo\images\val",
        "lab_dir": r"D:\SAR_Det_Research\hrsc2016_yolo\labels\val",
        "runs_csv": r"D:\SAR_Det_Research\hrsc2016_experiments\hrsc_runs.csv",
        "model_col": "variant",
        "runs_filter": {"dataset": "HRSC2016"},
        "runs_tags": {"baseline": "baseline", "dia": "dia_l2_k3"},
        "model_root": r"D:\SAR_Det_Research\hrsc2016_experiments",
        "model_prefix": "HRSC",
        "model_tagdir": {"baseline": "baseline", "dia": "dia_l2_k3"},
        "rep_seed": 42,
    },
]
CLASS_NAMES = ["ship"]
SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# ====================== 自定义模块（加载 DIA best.pt 的 pickle 依赖）======================
import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv


class StripConv(nn.Module):
    def __init__(self, c1, c2, k=3, stride=1, act=True):
        super().__init__()
        self.conv_h = nn.Conv2d(c1, c1, kernel_size=(1, k), stride=stride,
                                padding=(0, k // 2), groups=c1, bias=False)
        self.bn_h = nn.BatchNorm2d(c1)
        self.conv_v = nn.Conv2d(c1, c1, kernel_size=(k, 1), stride=stride,
                                padding=(k // 2, 0), groups=c1, bias=False)
        self.bn_v = nn.BatchNorm2d(c1)
        self.pw = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_pw = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        x = self.act(self.bn_h(self.conv_h(x)))
        x = self.act(self.bn_v(self.conv_v(x)))
        x = self.act(self.bn_pw(self.pw(x)))
        return x


class DualBranchBottleneck(nn.Module):
    def __init__(self, c, shortcut=True, k=3):
        super().__init__()
        self.shortcut = shortcut
        half_c = c // 2
        self.branch_origin = Conv(c, half_c, 3, 1)
        self.branch_strip = StripConv(c, half_c, k=k)

    def forward(self, x):
        x_origin = self.branch_origin(x)
        x_strip = self.branch_strip(x)
        out = torch.cat([x_origin, x_strip], dim=1)
        return x + out if self.shortcut else out


class C2f_DualBranch(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            DualBranchBottleneck(self.c, shortcut=shortcut, k=k)
            for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ====================== 统计函数（det 环境无 scipy，自含）======================
def mean(xs):
    return sum(xs) / len(xs)


def sample_std(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lnb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) \
        + a * math.log(x) + b * math.log(1 - x)
    bt = math.exp(lnb)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * betacf(a, b, x) / a
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b


def t_cdf(t, df):
    x = df / (df + t * t)
    ib = betai(df / 2.0, 0.5, x)
    return 1.0 - 0.5 * ib if t >= 0 else 0.5 * ib


def t_pvalue(t, df):
    return 2.0 * (1.0 - t_cdf(abs(t), df))


def t_ppf(p, df):
    """t 分布分位数（反 CDF），p in (0,1)，df>=1"""
    sign = 1.0 if p >= 0.5 else -1.0
    target = p if p >= 0.5 else 1.0 - p
    lo, hi = 0.0, 200.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < target:
            lo = mid
        else:
            hi = mid
    return sign * (lo + hi) / 2.0


# ====================== COCO 构造与指标 ======================
def coco_from_dict(d):
    from pycocotools.coco import COCO
    c = COCO()
    c.dataset = d
    c.createIndex()
    return c


def coco_stats(coco_gt_dict, dt_anns):
    """返回 dict(mkey->value)；dt_anns 为空则全 0"""
    from pycocotools.cocoeval import COCOeval
    empty = {k: 0.0 for k in MKEYS}
    if len(dt_anns) == 0:
        return empty
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            cg = coco_from_dict(coco_gt_dict)
            cd = cg.loadRes(dt_anns)
            ce = COCOeval(cg, cd, "bbox")
            ce.evaluate()
            ce.accumulate()
            ce.summarize()   # 填充 self.stats（打印被 redirect_stdout 吞掉）
            st = ce.stats
        return {"mAP50": float(st[1]), "mAP5095": float(st[0]),
                "APs": float(st[3]), "APm": float(st[4]), "APl": float(st[5])}
    except Exception as e:
        print("   [warn] coco eval failed:", e)
        return empty


def sorted_images(img_dir):
    p = Path(img_dir)
    return sorted([f for f in p.iterdir()
                   if f.suffix.lower() in SUFFIXES], key=lambda x: x.name)


# ---------- 阶段1：GT / det 落盘 ----------
def build_gt_json(img_paths, lab_dir):
    cats = [{"id": i, "name": n, "supercategory": "object"}
            for i, n in enumerate(CLASS_NAMES)]
    images, anns = [], []
    aid = 0
    for iid, ip in enumerate(img_paths):
        with Image.open(ip) as im:
            w, h = im.size
        images.append({"id": iid, "file_name": ip.name, "width": w, "height": h})
        lp = Path(lab_dir) / (ip.stem + ".txt")
        if lp.exists():
            with open(lp, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            for line in lines:
                p = line.split()
                cls = int(p[0])
                cx, cy, bw, bh = map(float, p[1:5])
                bw *= w
                bh *= h
                bx = cx * w - bw / 2.0
                by = cy * h - bh / 2.0
                anns.append({"id": aid, "image_id": iid, "category_id": cls,
                             "bbox": [round(bx, 3), round(by, 3),
                                      round(bw, 3), round(bh, 3)],
                             "area": round(bw * bh, 3), "iscrowd": 0})
                aid += 1
    return {"images": images, "annotations": anns, "categories": cats}


def build_det_json(weight_path, img_paths):
    from ultralytics import YOLO
    model = YOLO(str(weight_path))
    id_by_stem = {ip.stem: i for i, ip in enumerate(img_paths)}
    results = model.predict(source=[str(p) for p in img_paths],
                            imgsz=IMG_SIZE, conf=CONF_THRES, iou=IOU_THRES,
                            device=DEVICE, verbose=False)
    det = []
    did = 0
    for r in results:
        stem = Path(r.path).stem
        iid = id_by_stem.get(stem)
        if iid is None or r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = xyxy[i]
            det.append({"id": did, "image_id": int(iid),
                        "category_id": int(clss[i]),
                        "bbox": [round(float(x1), 3), round(float(y1), 3),
                                 round(float(x2 - x1), 3), round(float(y2 - y1), 3)],
                        "score": round(float(confs[i]), 5)})
            did += 1
    return det


def weight_path_for(cfg, key, seed):
    root = Path(cfg["model_root"])
    d = root / f"{cfg['model_prefix']}_{cfg['model_tagdir'][key]}_seed{seed}" / "weights" / "best.pt"
    return d


def stage1_prepare(cfg, ds_dir, img_paths):
    gt_path = ds_dir / "gt_coco.json"
    if gt_path.exists():
        gt = json.load(open(gt_path, "r", encoding="utf-8"))
    else:
        print("  生成 GT ...")
        gt = build_gt_json(img_paths, cfg["lab_dir"])
        json.dump(gt, open(gt_path, "w", encoding="utf-8"),
                  ensure_ascii=False)
    det_paths = {}
    for key in ["baseline", "dia"]:
        dp = ds_dir / f"det_{key}.json"
        det_paths[key] = dp
        if dp.exists():
            continue
        wp = weight_path_for(cfg, key, cfg["rep_seed"])
        if not wp.exists():
            print("  [warn] 缺权重:", wp)
            continue
        print(f"  推理 {key} (seed={cfg['rep_seed']}) ...")
        det = build_det_json(wp, img_paths)
        json.dump(det, open(dp, "w", encoding="utf-8"), ensure_ascii=False)
    return gt, det_paths


# ---------- 阶段2：图像级 bootstrap ----------
def run_bootstrap(gt, det_anns, sampled_idx, csv_out):
    """sampled_idx: B x N 矩阵（每行为一次有放回采样的 src 图像下标）"""
    if csv_out.exists():
        mat = np.loadtxt(csv_out, delimiter=",", skiprows=1)
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        return mat
    images = gt["images"]
    gt_by_img = {}
    for a in gt["annotations"]:
        gt_by_img.setdefault(a["image_id"], []).append(a)
    dt_by_img = {}
    for d in det_anns:
        dt_by_img.setdefault(d["image_id"], []).append(d)

    B, N = sampled_idx.shape
    rows = np.zeros((B, len(MKEYS)))
    for b in range(B):
        new_images, new_gt, new_dt = [], [], []
        ag = 0
        for slot in range(N):
            src = images[int(sampled_idx[b, slot])]
            nid = b * 1_000_000 + slot
            new_images.append({"id": nid, "file_name": src["file_name"],
                               "width": src["width"], "height": src["height"]})
            for a in gt_by_img.get(src["id"], []):
                na = dict(a)
                na["id"] = ag
                na["image_id"] = nid
                new_gt.append(na)
                ag += 1
            for d in dt_by_img.get(src["id"], []):
                nd = dict(d)
                nd["image_id"] = nid
                nd.pop("id", None)
                new_dt.append(nd)
        gt_dict = {"images": new_images, "annotations": new_gt,
                   "categories": gt["categories"]}
        stats = coco_stats(gt_dict, new_dt)
        rows[b] = [stats[k] for k in MKEYS]
        if (b + 1) % 50 == 0:
            print(f"    bootstrap {b + 1}/{B}")
    np.savetxt(csv_out, rows, delimiter=",", header=",".join(MKEYS),
               comments="", fmt="%.6f")
    return rows


# ---------- 阶段3：读 runs CSV，噪声地板 / 配对 ----------
def read_runs(cfg):
    rows = []
    with open(cfg["runs_csv"], "r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            ok = all(str(r.get(k)) == str(v) for k, v in cfg["runs_filter"].items())
            if ok:
                rows.append(r)
    return rows


def group_runs(cfg, rows, analyze_key):
    """返回 {seed: {mkey: value}}"""
    tag = cfg["runs_tags"][analyze_key]
    out = {}
    for r in rows:
        if r[cfg["model_col"]] == tag:
            seed = r["seed"]
            out[int(float(seed))] = {k: float(r[MCOL[k]]) for k in MKEYS}
    return out


def stage3_tables(cfg, gt, boot_mat, out_tables, ds_name):
    rows = read_runs(cfg)
    base = group_runs(cfg, rows, "baseline")
    dia = group_runs(cfg, rows, "dia")
    n_run = min(len(base), len(dia))

    # 3.1 运行间噪声地板
    floor_rows = []
    for key, grp in [("baseline", base), ("dia", dia)]:
        rec = {"dataset": ds_name, "variant": key, "n": len(grp)}
        for k in MKEYS:
            vals = [grp[s][k] for s in grp]
            rec[k + "_mean"] = round(mean(vals), 4)
            rec[k + "_std"] = round(sample_std(vals), 4)
        floor_rows.append(rec)

    # 3.2 bootstrap（测试集抽样）噪声
    boot_rows = []
    brec = {"dataset": ds_name, "model": "rep_seed%d" % cfg["rep_seed"], "B": boot_mat.shape[0]}
    for j, k in enumerate(MKEYS):
        v = np.asarray(boot_mat[:, j], dtype=float)
        vv = v[v > -0.5]                       # 去掉 pycocotools 的 -1 缺失标记
        brec[k + "_validB"] = int(len(vv))
        if len(vv) >= 2:
            brec[k + "_mean"] = round(float(np.mean(vv)), 4)
            brec[k + "_bootstd"] = round(float(np.std(vv, ddof=1)), 4)
            lo, hi = np.percentile(vv, [2.5, 97.5])
            brec[k + "_ci95"] = f"[{round(float(lo),4)},{round(float(hi),4)}]"
        elif len(vv) == 1:
            brec[k + "_mean"] = round(float(vv[0]), 4)
            brec[k + "_bootstd"] = 0.0
            brec[k + "_ci95"] = "NA"
        else:
            brec[k + "_mean"] = "NA"
            brec[k + "_bootstd"] = "NA"
            brec[k + "_ci95"] = "NA"
    boot_rows.append(brec)

    # 3.3 配对差
    common = sorted(set(base) & set(dia))
    paired_rows = []
    sd_by_metric = {}
    for k in MKEYS:
        d = [dia[s][k] - base[s][k] for s in common]
        md = mean(d)
        sd = sample_std(d)
        sd_by_metric[k] = sd
        n = len(d)
        if sd > 0 and n > 1:
            t = md / (sd / math.sqrt(n))
            p = t_pvalue(t, n - 1)
        else:
            t, p = 0.0, 1.0
        paired_rows.append({"dataset": ds_name, "metric": k, "n": n,
                            "mean_diff": round(md, 4),
                            "pairdiff_std": round(sd, 4),
                            "t": round(t, 3), "p": round(p, 4)})

    # 3.4 MDE
    mde_rows = []
    base_std = {k: sample_std([base[s][k] for s in base]) for k in MKEYS}
    dia_std = {k: sample_std([dia[s][k] for s in dia]) for k in MKEYS}
    df = n_run - 1
    for k in MKEYS:
        se_single = math.sqrt(base_std[k] ** 2 + dia_std[k] ** 2)
        sd = sd_by_metric[k]
        tcrit = t_ppf(1 - ALPHA / 2, df)
        tpow = t_ppf(POWER, df)
        mde_rows.append({
            "dataset": ds_name, "metric": k, "n": n_run,
            "single_MDE90": round(Z90 * se_single, 4),
            "single_MDE95": round(Z95 * se_single, 4),
            "paired_MDE_sig": round(tcrit * sd / math.sqrt(n_run), 4),
            "paired_MDE_power80": round((tcrit + tpow) * sd / math.sqrt(n_run), 4),
        })

    # 3.5 所需种子数（配对）
    req_rows = []
    targets = [0.005, 0.01, 0.02, 0.03, 0.05]
    for k in MKEYS:
        sd = sd_by_metric[k]
        n_for = {}
        if sd > 0:
            for dtarget in targets:
                found = ""
                for n in range(2, 401):
                    tc = t_ppf(1 - ALPHA / 2, n - 1)
                    tb = t_ppf(POWER, n - 1)
                    if (tc + tb) * sd / math.sqrt(n) <= dtarget:
                        found = n
                        break
                n_for[str(dtarget)] = found
        rec = {"dataset": ds_name, "metric": k, "pairdiff_std": round(sd, 4)}
        for dtarget in targets:
            rec[f"n_for_d{dtarget}"] = n_for.get(str(dtarget), "")
        req_rows.append(rec)

    # std 估计相对精度 -> n
    stdprec = []
    for r in [0.20, 0.25, 0.30]:
        nstd = int(math.ceil(1 + 0.5 * (Z95 / r) ** 2))
        stdprec.append({"target_rel_precision": r, "required_n": nstd})

    return {
        "floor": (floor_rows, ["dataset", "variant", "n"]
                  + [k + "_" + s for k in MKEYS for s in ("mean", "std")]),
        "boot": (boot_rows, ["dataset", "model", "B"]
                 + [k + "_validB" for k in MKEYS]
                 + [k + "_mean" for k in MKEYS]
                 + [k + "_bootstd" for k in MKEYS]
                 + [k + "_ci95" for k in MKEYS]),
        "paired": (paired_rows, ["dataset", "metric", "n", "mean_diff",
                                 "pairdiff_std", "t", "p"]),
        "mde": (mde_rows, ["dataset", "metric", "n", "single_MDE90",
                           "single_MDE95", "paired_MDE_sig", "paired_MDE_power80"]),
        "reqn": (req_rows, ["dataset", "metric", "pairdiff_std"]
                 + [f"n_for_d{t}" for t in targets]),
        "stdprec": (stdprec, ["target_rel_precision", "required_n"]),
    }


def write_csv(path, data, header):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in data:
            w.writerow({k: r.get(k, "") for k in header})


# ---------- 阶段4：两层噪声图 ----------
def make_two_layer_figure(all_floor, all_boot, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Times New Roman"

    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return 0.0

    focus = ["mAP5095", "APl"]
    ds_names = [d["name"] for d in DATASETS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=300)
    x = np.arange(len(ds_names))
    width = 0.35
    colors_run = {"mAP5095": "#1f4e79", "APl": "#2e6f8f"}
    colors_boot = {"mAP5095": "#b4502a", "APl": "#d89a7a"}
    for ax, k in zip(axes, focus):
        run_vals, boot_vals = [], []
        for ds in ds_names:
            fr = [r for r in all_floor if r["dataset"] == ds and r["variant"] == "baseline"][0]
            br = [r for r in all_boot if r["dataset"] == ds][0]
            run_vals.append(_f(fr[k + "_std"]))
            boot_vals.append(_f(br[k + "_bootstd"]))
        ax.bar(x - width / 2, run_vals, width, label="Run-to-run (training)",
               color=colors_run[k], edgecolor="black", linewidth=.6)
        ax.bar(x + width / 2, boot_vals, width, label="Test-set bootstrap",
               color=colors_boot[k], edgecolor="black", linewidth=.6)
        ax.set_xticks(x)
        ax.set_xticklabels(ds_names)
        ax.set_ylabel("Standard deviation")
        ax.set_title(k, fontsize=12)
        ax.grid(axis="y", alpha=.3, linestyle="--")
        ax.legend(fontsize=9)
    fig.suptitle("Two Layers of Metric Noise: Training Stochasticity vs Test-Set Sampling",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ====================== 主流程 ======================
def main():
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    out_root = OUT_SMOKE if smoke else OUT_FULL
    B = 3 if smoke else BOOT_B
    ds_list = DATASETS[:1] if smoke else DATASETS
    out_root.mkdir(parents=True, exist_ok=True)
    tables_dir = out_root / "tables"
    tables_dir.mkdir(exist_ok=True)

    tnames = {"floor": "noise_floor.csv", "boot": "test_bootstrap_noise.csv",
              "paired": "paired_diffs.csv", "mde": "mde_table.csv",
              "reqn": "required_n.csv", "stdprec": "std_precision_n.csv"}
    merged = {k: {"data": [], "header": None} for k in tnames}
    for cfg in ds_list:
        ds_name = cfg["name"]
        ds_dir = out_root / ds_name
        ds_dir.mkdir(exist_ok=True)
        print("=" * 64)
        print("数据集:", ds_name)
        img_paths = sorted_images(cfg["img_dir"])
        if smoke:
            img_paths = img_paths[:40]
        print("  图像数:", len(img_paths))

        gt, det_paths = stage1_prepare(cfg, ds_dir, img_paths)

        # bootstrap 采样矩阵（baseline/dia 共用，配对）
        N = len(img_paths)
        rng = np.random.default_rng(BOOT_SEED)
        sampled_idx = np.vstack([rng.integers(0, N, size=N) for _ in range(B)])

        boot_by_key = {}
        for key in ["baseline", "dia"]:
            print(f"  bootstrap {key} (B={B}) ...")
            det = json.load(open(det_paths[key], "r", encoding="utf-8"))
            mat = run_bootstrap(gt, det, sampled_idx,
                                ds_dir / f"boot_{key}.csv")
            boot_by_key[key] = mat

        tables = stage3_tables(cfg, gt, boot_by_key["baseline"],
                               tables_dir, ds_name)
        for tkey, fname in tnames.items():
            data, header = tables[tkey]
            write_csv(ds_dir / fname, data, header)          # 每数据集一份
            merged[tkey]["data"].extend(data)
            if merged[tkey]["header"] is None:
                merged[tkey]["header"] = header

        # 控制台速览
        print("\n  --- 配对差（dia-baseline）---")
        for r in tables["paired"][0]:
            print(f"   {r['metric']:<8} mean={r['mean_diff']:+.4f} "
                  f"s_d={r['pairdiff_std']:.4f} p={r['p']}")
        print("  --- MDE（单次95 / 配对sig / 配对power80）---")
        for r in tables["mde"][0]:
            print(f"   {r['metric']:<8} {r['single_MDE95']:.4f} / "
                  f"{r['paired_MDE_sig']:.4f} / {r['paired_MDE_power80']:.4f}")

    # 全量：合并总表 + 两层噪声图
    if not smoke:
        for tkey, fname in tnames.items():
            write_csv(tables_dir / fname,
                      merged[tkey]["data"], merged[tkey]["header"])
        all_floor = merged["floor"]["data"]
        all_boot = merged["boot"]["data"]
        fig_png = out_root / "Figure_two_layer_noise.png"
        make_two_layer_figure(all_floor, all_boot, fig_png)
        print("\n两层噪声图:", fig_png)

    print("\n全部完成。输出目录:", out_root)


if __name__ == "__main__":
    main()
