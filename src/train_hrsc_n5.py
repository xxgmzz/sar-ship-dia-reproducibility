# -*- coding: utf-8 -*-
"""
HRSC2016 补跑：3 种子 -> 5 种子（仅 baseline + dia_l2_k3）
=========================================================
回应审稿人"HRSC2016 仅 3 跑、负向显著性不稳，建议 ≥5"的意见。
在已有 3 种子（42/123/2024）基础上，为 baseline 和 dia_l2_k3 各补 2 个
种子（7/2025，与 HRSID 主线种子一致），共 4 次训练；断点续跑自动跳过已完成。
ALK-DIA 非主线（已被证伪的变体），保持原 3 跑，不补。
训练后输出独立的 n=5 统计与配对 t，不覆盖原 n=3 文件。
协议与 run_hrsc2016_experiments.py 完全一致：yolov8n.pt 加载、动态替换、COCO 评估。
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
SEEDS = [42, 123, 2024, 7, 2025]
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
DATA_YAML = r"D:\SAR_Det_Research\hrsc2016_yolo\hrsc2016.yaml"
VAL_IMG = Path(r"D:\SAR_Det_Research\hrsc2016_yolo\images\val")
VAL_LABEL = Path(r"D:\SAR_Det_Research\hrsc2016_yolo\labels\val")
OUT_ROOT = Path(r"D:\SAR_Det_Research\hrsc2016_experiments")

METRIC_KEYS = ["mAP@0.5", "mAP@0.5:0.95", "AP-small", "AP-medium", "AP-large"]


# ====================== DIA 模块（与 HRSC 原脚本一致）======================
class StripConv(nn.Module):
    def __init__(self, c1, c2, k=3, stride=1, act=True):
        super().__init__()
        self.conv_h = nn.Conv2d(c1, c1, (1, k), stride, (0, k // 2), groups=c1, bias=False)
        self.bn_h = nn.BatchNorm2d(c1)
        self.conv_v = nn.Conv2d(c1, c1, (k, 1), stride, (k // 2, 0), groups=c1, bias=False)
        self.bn_v = nn.BatchNorm2d(c1)
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn_pw = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        x = self.act(self.bn_h(self.conv_h(x)))
        x = self.act(self.bn_v(self.conv_v(x)))
        return self.act(self.bn_pw(self.pw(x)))


class DualBranchBottleneck(nn.Module):
    def __init__(self, c, shortcut=True, k=3):
        super().__init__()
        self.shortcut = shortcut
        half = c // 2
        self.branch_origin = Conv(c, half, 3, 1)
        self.branch_strip = StripConv(c, half, k=k)

    def forward(self, x):
        out = torch.cat([self.branch_origin(x), self.branch_strip(x)], dim=1)
        return x + out if self.shortcut else out


class C2f_DualBranch(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(DualBranchBottleneck(self.c, shortcut, k) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def make_dia_trainer(layers, k):
    class _T(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg, weights, verbose)
            print("\n" + "=" * 70 + f"\n动态替换 -> C2f_DualBranch：layers={layers}, k={k}")
            for idx in layers:
                o = model.model[idx]
                c1 = o.cv1.conv.in_channels; c2 = o.cv2.conv.out_channels
                n = len(o.m); shortcut = o.m[0].add; e = o.c / c2
                new = C2f_DualBranch(c1, c2, n, shortcut, e=e, k=k)
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


def run_one(variant, seed):
    name = f"HRSC_{variant}_seed{seed}"
    exp_dir = OUT_ROOT / name
    metrics_csv = exp_dir / "val_coco_result" / "val_metrics.csv"
    rec = {"dataset": "HRSC2016", "variant": variant, "seed": seed}
    if metrics_csv.exists():
        row = list(csv.DictReader(open(metrics_csv, encoding="utf-8-sig")))[0]
        for k in METRIC_KEYS:
            rec[k] = row.get(k)
        print(f"[跳过-已完成] {name}")
        return rec
    print("\n" + "#" * 70 + f"\n# 训练 {name}\n" + "#" * 70)
    trainer = make_dia_trainer(REPLACE_LAYERS, K) if variant == "dia_l2_k3" else None
    model = YOLO(INIT_WEIGHTS)
    model.train(data=DATA_YAML, imgsz=IMG_SIZE, batch=BATCH_SIZE, epochs=EPOCHS,
                device=DEVICE, seed=seed, mosaic=MOSAIC, workers=WORKERS,
                project=str(OUT_ROOT), name=name, exist_ok=True,
                trainer=trainer, verbose=True)
    metrics = evaluate(exp_dir); rec.update(metrics)
    with open(metrics_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(METRIC_KEYS); w.writerow([metrics[k] for k in METRIC_KEYS])
    print(f"  已保存: {metrics_csv}")
    return rec


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


def paired_ttest(base, other):
    n = len(base); df = n - 1
    out = {}
    for k in METRIC_KEYS:
        d = np.array([float(x[k]) for x in other]) - np.array([float(x[k]) for x in base])
        t = d.mean() / (d.std(ddof=1) / math.sqrt(n))
        p = betai(df / 2, 0.5, df / (df + t * t))
        out[k] = {"mean_diff": round(float(d.mean()), 4), "s_d": round(float(d.std(ddof=1)), 4),
                  "t": round(float(t), 3), "p": round(float(p), 4), "diffs": np.round(d, 4).tolist()}
    return out


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    groups = {"baseline": [], "dia_l2_k3": []}
    for variant in ("baseline", "dia_l2_k3"):
        for seed in SEEDS:
            try:
                groups[variant].append(run_one(variant, seed))
            except Exception as e:
                import traceback
                print(f"[失败] {variant}/seed{seed}: {e}"); traceback.print_exc()

    # runs n5
    fields = ["dataset", "variant", "seed"] + METRIC_KEYS
    with open(OUT_ROOT / "hrsc_runs_n5.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fields); w.writeheader()
        for v in ("baseline", "dia_l2_k3"):
            for r in groups[v]:
                w.writerow(r)

    def meanstd(recs):
        r = {}
        for k in METRIC_KEYS:
            arr = np.array([float(x[k]) for x in recs])
            r[k] = (round(arr.mean(), 4), round(arr.std(ddof=1), 4))
        return r
    ms = {v: meanstd(groups[v]) for v in groups}
    with open(OUT_ROOT / "hrsc_mean_std_n5.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["variant", "n"] + [x for k in METRIC_KEYS for x in (k + "_mean", k + "_std")])
        for v in ("baseline", "dia_l2_k3"):
            w.writerow([v, len(groups[v])] + [x for k in METRIC_KEYS for x in ms[v][k]])

    tt = paired_ttest(groups["baseline"], groups["dia_l2_k3"])
    with open(OUT_ROOT / "hrsc_ttest_n5.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "metric", "mean_diff", "s_d", "t", "p_two_sided", "per_seed_diffs"])
        for k in METRIC_KEYS:
            w.writerow(["dia_vs_base_n5", k, tt[k]["mean_diff"], tt[k]["s_d"], tt[k]["t"],
                        tt[k]["p"], tt[k]["diffs"]])

    print("\n" + "=" * 92)
    print(f"HRSC2016 dia vs baseline（n={len(SEEDS)}）")
    print("-" * 92)
    print(f"{'metric':<16}{'baseline':>17}{'dia':>17}{'mean diff':>10}{'p':>8}")
    for k in METRIC_KEYS:
        bs = f"{ms['baseline'][k][0]} ± {ms['baseline'][k][1]}"
        cs = f"{ms['dia_l2_k3'][k][0]} ± {ms['dia_l2_k3'][k][1]}"
        print(f"{k:<16}{bs:>17}{cs:>17}{tt[k]['mean_diff']:>10}{tt[k]['p']:>8}")
    print("\n" + "=" * 92)
    print(f"runs:     {OUT_ROOT / 'hrsc_runs_n5.csv'}")
    print(f"mean±std: {OUT_ROOT / 'hrsc_mean_std_n5.csv'}")
    print(f"t检验:    {OUT_ROOT / 'hrsc_ttest_n5.csv'}")


if __name__ == "__main__":
    main()
