# -*- coding: utf-8 -*-
"""
ISO 各向同性对照实验（HRSID，n=8）
=========================================================
目的（回应两位审稿人共同指出的关键缺失对照）：
  DIA-Conv 第二分支是 depthwise 1×3 + 3×1 十字条带（各向异性）；
  本对照把第二分支换成"完整 depthwise 3×3"（各向同性），其余结构完全相同
  （第一分支标准 3×3、shortcut、BN、SiLU、仅注入第 2 层、channel 1:1）。
  已用 verify_iso_param_align.py 核实：DIA 与 ISO 全模型仅差 16 个参数（+0.001%），
  MACs 4.3477G vs 4.3473G，属严格 matched-parameter 对照。

  判决逻辑：
    - 若 DIA 显著优于 ISO → 增益可归因于"各向异性条带"，anisotropic 标题词坐实；
    - 若 ISO 与 DIA 无差异（或 ISO 更好）→ 增益来自 depthwise/双分支结构本身，
      而非各向异性，论文据实弱化机制归因（符合全文证伪风格）。

  与主线 HRSID n=8 完全相同的 8 个种子（42/123/2024/7/2025/11/88/314），
  训练后按 seed 配对，读 multiseed_experiments 里已有 baseline/dia 结果，
  输出三组配对 t：dia-base、iso-base、dia-iso。
  不依赖 yaml、从 yolov8n.pt 加载、动态替换、COCO 评估、断点续跑。
"""
import csv
import json
import math
from pathlib import Path
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import Conv
from ultralytics.models.yolo.detect import DetectionTrainer
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ====================== 全局配置 ======================
SEEDS = [42, 123, 2024, 7, 2025, 11, 88, 314]
IMG_SIZE = 640
BATCH_SIZE = 8
EPOCHS = 200
DEVICE = 0
MOSAIC = 0
WORKERS = 0
CONF_THRES = 0.001
IOU_THRES = 0.6
K = 3
REPLACE_LAYERS = [2]

INIT_WEIGHTS = r"D:\SAR_Det_Research\exp00_baseline_yolov8n_ssdd\yolov8n.pt"
DATA_YAML = r"D:\SAR_Det_Research\hrsid_experiments\configs\hrsid.yaml"
VAL_IMG = Path(r"D:\SAR_Det_Research\hrsid_experiments\data\images\val")
VAL_LABEL = Path(r"D:\SAR_Det_Research\hrsid_experiments\data\labels\val")

OUT_ROOT = Path(r"D:\SAR_Det_Research\iso_control_experiments")
MAIN_ROOT = Path(r"D:\SAR_Det_Research\multiseed_experiments")

METRIC_KEYS = ["mAP@0.5", "mAP@0.5:0.95", "AP-small", "AP-medium", "AP-large"]


# ====================== ISO 模块（自检已对齐 DIA）======================
class IsoConv(nn.Module):
    """各向同性 depthwise-separable：完整 depthwise k×k + pointwise 1×1。
    与主线 StripConv（1×k + k×1 十字）唯一差别为空间核形状。"""
    def __init__(self, c1, c2, k=3, stride=1, act=True):
        super().__init__()
        self.conv_dw = nn.Conv2d(c1, c1, k, stride, k // 2, groups=c1, bias=False)
        self.bn_dw = nn.BatchNorm2d(c1)
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn_pw = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        x = self.act(self.bn_dw(self.conv_dw(x)))
        return self.act(self.bn_pw(self.pw(x)))


class IsoDualBranchBottleneck(nn.Module):
    def __init__(self, c, shortcut=True, k=3):
        super().__init__()
        self.shortcut = shortcut
        half = c // 2
        self.branch_origin = Conv(c, half, 3, 1)
        self.branch_iso = IsoConv(c, half, k=k)

    def forward(self, x):
        out = torch.cat([self.branch_origin(x), self.branch_iso(x)], dim=1)
        return x + out if self.shortcut else out


class C2f_Iso(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(IsoDualBranchBottleneck(self.c, shortcut, k) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def make_iso_trainer(layers, k):
    class _T(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg, weights, verbose)
            print("\n" + "=" * 70 + f"\n动态替换 -> C2f_Iso（各向同性 depthwise 3×3 对照）：layers={layers}, k={k}")
            for idx in layers:
                o = model.model[idx]
                c1 = o.cv1.conv.in_channels; c2 = o.cv2.conv.out_channels
                n = len(o.m); shortcut = o.m[0].add; e = o.c / c2
                new = C2f_Iso(c1, c2, n, shortcut, e=e, k=k)
                new.f, new.i = o.f, o.i
                model.model[idx] = new
                print(f"  第{idx}层完成：{c1}->{c2}, n={n}, k={k}")
            print(f"  总参数量：{sum(p.numel() for p in model.parameters()):,}\n" + "=" * 70)
            return model
    return _T


# ====================== COCO 评估 ======================
def get_sorted_img_list(img_dir):
    suffixes = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    return sorted([f for f in img_dir.iterdir() if f.suffix.lower() in suffixes])


def generate_coco_gt(img_list, label_dir):
    coco_gt = {"images": [], "annotations": [],
               "categories": [{"id": 0, "name": "ship", "supercategory": "object"}]}
    ann_id = 0
    for img_id, img_path in enumerate(img_list):
        with Image.open(img_path) as img:
            w, h = img.size
        coco_gt["images"].append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})
        lp = label_dir / (img_path.stem + ".txt")
        if not lp.exists():
            continue
        for line in [l for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]:
            p = line.split()
            cls_id = int(p[0]); cx, cy, bw, bh = map(float, p[1:5])
            bw *= w; bh *= h; bx = cx * w - bw / 2; by = cy * h - bh / 2
            coco_gt["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": cls_id,
                "bbox": [round(bx, 3), round(by, 3), round(bw, 3), round(bh, 3)],
                "area": round(bw * bh, 3), "iscrowd": 0})
            ann_id += 1
    return coco_gt


def generate_coco_predictions(img_list, model):
    preds = []
    for img_id, img_path in enumerate(img_list):
        r = model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=CONF_THRES,
                          iou=IOU_THRES, device=DEVICE, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy(); confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = xyxy[i]
            preds.append({"image_id": img_id, "category_id": int(clss[i]),
                          "bbox": [round(float(x1), 3), round(float(y1), 3),
                                   round(float(x2 - x1), 3), round(float(y2 - y1), 3)],
                          "score": round(float(confs[i]), 5)})
    return preds


def evaluate(exp_dir):
    res_dir = exp_dir / "val_coco_result"; res_dir.mkdir(parents=True, exist_ok=True)
    img_list = get_sorted_img_list(VAL_IMG)
    print(f"  验证集图片数: {len(img_list)}；生成真值...")
    gt = generate_coco_gt(img_list, VAL_LABEL)
    model = YOLO(str(exp_dir / "weights" / "best.pt"))
    pred = generate_coco_predictions(img_list, model)
    print(f"  预测框总数: {len(pred)}")
    gt_p = res_dir / "val_gt_coco.json"; pr_p = res_dir / "val_predictions.json"
    gt_p.write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")
    pr_p.write_text(json.dumps(pred, ensure_ascii=False, indent=2), encoding="utf-8")
    coco_gt = COCO(str(gt_p)); coco_dt = coco_gt.loadRes(str(pr_p))
    ce = COCOeval(coco_gt, coco_dt, "bbox"); ce.evaluate(); ce.accumulate(); ce.summarize()
    s = ce.stats
    return {"mAP@0.5": round(float(s[1]), 4), "mAP@0.5:0.95": round(float(s[0]), 4),
            "AP-small": round(float(s[3]), 4), "AP-medium": round(float(s[4]), 4),
            "AP-large": round(float(s[5]), 4)}


# ====================== 训练一个种子（断点续跑）======================
def run_one(seed):
    name = f"HRSID_iso_l2_k3_seed{seed}"
    exp_dir = OUT_ROOT / name
    metrics_csv = exp_dir / "val_coco_result" / "val_metrics.csv"
    rec = {"dataset": "HRSID", "variant": "iso_l2_k3", "seed": seed}
    if metrics_csv.exists():
        row = list(csv.DictReader(open(metrics_csv, encoding="utf-8-sig")))[0]
        for k in METRIC_KEYS:
            rec[k] = row.get(k)
        print(f"[跳过-已完成] {name}")
        return rec
    print("\n" + "#" * 70 + f"\n# 训练 {name}\n" + "#" * 70)
    model = YOLO(INIT_WEIGHTS)
    trainer = make_iso_trainer(REPLACE_LAYERS, K)
    model.train(data=DATA_YAML, imgsz=IMG_SIZE, batch=BATCH_SIZE, epochs=EPOCHS,
                device=DEVICE, seed=seed, mosaic=MOSAIC, workers=WORKERS,
                project=str(OUT_ROOT), name=name, exist_ok=True,
                trainer=trainer, verbose=True)
    metrics = evaluate(exp_dir); rec.update(metrics)
    with open(metrics_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(METRIC_KEYS); w.writerow([metrics[k] for k in METRIC_KEYS])
    print(f"  已保存: {metrics_csv}")
    return rec


# ====================== 读已有 baseline/dia（按 seed 配对）======================
def read_existing(variant_dir_pat):
    by_seed = {}
    for seed in SEEDS:
        p = MAIN_ROOT / f"{variant_dir_pat}{seed}" / "val_coco_result" / "val_metrics.csv"
        if p.exists():
            row = list(csv.DictReader(open(p, encoding="utf-8-sig")))[0]
            by_seed[seed] = {k: float(row[k]) for k in METRIC_KEYS}
    return by_seed


# ====================== 配对 t（自包含，无 scipy）======================
def betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c = 1.; d = 1. - qab * x / qap
    if abs(d) < FPMIN: d = FPMIN
    d = 1. / d; h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2)); d = 1. + aa * d
        if abs(d) < FPMIN: d = FPMIN
        c = 1. + aa / c
        if abs(c) < FPMIN: c = FPMIN
        d = 1. / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)); d = 1. + aa * d
        if abs(d) < FPMIN: d = FPMIN
        c = 1. + aa / c
        if abs(c) < FPMIN: c = FPMIN
        d = 1. / d; de = d * c
        if abs(de - 1.) < EPS: break
        h *= de
    return h


def betai(a, b, x):
    if x <= 0: return 0.
    if x >= 1: return 1.
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2): return bt * betacf(a, b, x) / a
    return 1 - bt * betacf(b, a, 1 - x) / b


def paired_ttest(a, b):
    """b - a，a/b 为 seed -> metrics dict"""
    n = len(a); df = n - 1
    out = {}
    for k in METRIC_KEYS:
        d = np.array([b[s][k] - a[s][k] for s in sorted(a)])
        t = d.mean() / (d.std(ddof=1) / math.sqrt(n))
        p = betai(df / 2, 0.5, df / (df + t * t))
        out[k] = {"mean_diff": round(float(d.mean()), 4), "s_d": round(float(d.std(ddof=1)), 4),
                  "t": round(float(t), 3), "p": round(float(p), 4), "diffs": np.round(d, 4).tolist()}
    return out


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    iso_recs = []
    for seed in SEEDS:
        try:
            iso_recs.append(run_one(seed))
        except Exception as e:
            import traceback
            print(f"[失败] iso/seed{seed}: {e}"); traceback.print_exc()

    # iso runs.csv
    iso_by_seed = {}
    for r in iso_recs:
        if r.get(METRIC_KEYS[0]) not in (None, ""):
            iso_by_seed[r["seed"]] = {k: float(r[k]) for k in METRIC_KEYS}
    fields = ["dataset", "variant", "seed"] + METRIC_KEYS
    with open(OUT_ROOT / "iso_runs.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fields); w.writeheader()
        for r in iso_recs:
            w.writerow(r)

    # iso mean±std
    def meanstd(by_seed):
        seeds = sorted(by_seed); out = {"n": len(seeds)}
        for k in METRIC_KEYS:
            arr = np.array([by_seed[s][k] for s in seeds])
            out[k + "_mean"] = round(float(arr.mean()), 4)
            out[k + "_std"] = round(float(arr.std(ddof=1)), 4) if len(seeds) > 1 else ""
        return out
    ms = {"iso_l2_k3": meanstd(iso_by_seed)}
    with open(OUT_ROOT / "iso_mean_std.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["variant", "n"] + [x for k in METRIC_KEYS for x in (k + "_mean", k + "_std")])
        row = ms["iso_l2_k3"]
        w.writerow(["iso_l2_k3", row["n"]] + [x for k in METRIC_KEYS for x in (row[k + "_mean"], row[k + "_std"])])

    # 读已有 baseline/dia
    base = read_existing("HRSID_baseline_seed")
    dia = read_existing("HRSID_dia_l2_k3_seed")
    common = sorted(set(base) & set(dia) & set(iso_by_seed))
    base = {s: base[s] for s in common}; dia = {s: dia[s] for s in common}; iso = {s: iso_by_seed[s] for s in common}
    print(f"\n配对种子数（base/dia/iso 三方共有）: {len(common)} -> {common}")

    tt = {"dia_minus_base": paired_ttest(base, dia),
          "iso_minus_base": paired_ttest(base, iso),
          "dia_minus_iso": paired_ttest(iso, dia)}
    with open(OUT_ROOT / "iso_control_ttest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "metric", "mean_diff", "s_d", "t", "p_two_sided", "per_seed_diffs"])
        for cmp_, d in tt.items():
            for k in METRIC_KEYS:
                w.writerow([cmp_, k, d[k]["mean_diff"], d[k]["s_d"], d[k]["t"], d[k]["p"], d[k]["diffs"]])

    print("\n" + "=" * 92)
    for cmp_, d in tt.items():
        print(f"\n{cmp_} (n={len(common)})")
        print("-" * 92)
        print(f"{'metric':<16}{'mean_diff':>10}{'s_d':>9}{'t':>9}{'p':>9}")
        for k in METRIC_KEYS:
            print(f"{k:<16}{d[k]['mean_diff']:>10}{d[k]['s_d']:>9}{d[k]['t']:>9}{d[k]['p']:>9}")
    print("\n" + "=" * 92)
    print(f"iso runs:    {OUT_ROOT / 'iso_runs.csv'}")
    print(f"mean±std:    {OUT_ROOT / 'iso_mean_std.csv'}")
    print(f"配对 t 检验: {OUT_ROOT / 'iso_control_ttest.csv'}")


if __name__ == "__main__":
    main()
