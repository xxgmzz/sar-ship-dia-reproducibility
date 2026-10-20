# -*- coding: utf-8 -*-
"""
SFW-YOLO 核心模块忠实复现 + 5 种子配对审计（B 路线）
====================================================
严格按 SFW 原文（DOI 10.1109/JMASS.2026.3681819）复现其 backbone 的
C2f_StripCGLU（Strip Attention + Convolutional Gated Linear Unit），
仅替换 backbone 前两个原生 C2f（YOLOv8 索引 2、4），对齐原文 Table III 的 B1
（原文 B1 = 2.98M / 7.8G；本实现替换后 = 2.99M，已核对）。

审计范围（论文中明确声明）：
  我们只复现并审计 C2f_StripCGLU 这一 strip-gated 核心；
  原文的 C2f_SFA（DCT 频率）、WRFF（宽感受野融合）、WIoUv3 损失不在本审计范围内。

配对设计：与 8 种子 baseline 的 5 个共同种子（42/123/2024/7/2025）逐一配对，
报 mean_diff、配对差值 std、t、p（betai，无 scipy）。

输出目录：D:\\SAR_Det_Research\\stripcglu_audit_experiments
  - HRSID_stripcglu_seed{seed}/weights/best.pt（5 个模型）
  - stripcglu_runs.csv（5 次单次指标）
  - stripcglu_mean_std.csv（mean±std）
  - stripcglu_vs_baseline_ttest.csv（配对检验）
建议脚本名：run_stripcglu_audit_5seed.py
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
from ultralytics.nn.modules.block import C2f
from ultralytics.models.yolo.detect import DetectionTrainer
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ====================== 配置 ======================
SEEDS = [42, 123, 2024, 7, 2025]          # 与 baseline 共同的 5 个种子
INIT_WEIGHTS = r"D:\SAR_Det_Research\exp00_baseline_yolov8n_ssdd\yolov8n.pt"
DATA_YAML = r"D:\SAR_Det_Research\hrsid_experiments\configs\hrsid.yaml"
VAL_IMG_DIR = Path(r"D:\SAR_Det_Research\hrsid_experiments\data\images\val")
VAL_LABEL_DIR = Path(r"D:\SAR_Det_Research\hrsid_experiments\data\labels\val")
# baseline 共同种子结果（配对真值来源）
BASELINE_RUN_ROOT = Path(r"D:\SAR_Det_Research\multiseed_experiments")
OUT = Path(r"D:\SAR_Det_Research\stripcglu_audit_experiments")

K = 7                 # 条带核长度（原文未写死，采用 k=7）
DROPOUT = 0.1
IMG_SIZE = 640
BATCH = 8
EPOCHS = 200
DEVICE = 0
MOSAIC = 0
WORKERS = 0
CONF_THRES = 0.001
IOU_THRES = 0.6
METRIC_KEYS = ["mAP@0.5", "mAP@0.5:0.95", "AP-small", "AP-medium", "AP-large"]


# ====================== 模块（忠实复现）======================
class StripAttention(nn.Module):
    """Strip Block：FC->GELU->Square(5x5 dw)->H Strip->V Strip->PW(出 attention A)->A⊗X1->FC。
    返回 (out, A)，A 供 CGLU 门控。"""
    def __init__(self, c, k=7):
        super().__init__()
        self.fc1 = Conv(c, c, 1, 1, act=nn.GELU())
        self.square = nn.Conv2d(c, c, 5, 1, 2, groups=c, bias=False)
        self.bn_sq = nn.BatchNorm2d(c)
        self.h = nn.Conv2d(c, c, (1, k), 1, (0, k // 2), groups=c, bias=False)
        self.v = nn.Conv2d(c, c, (k, 1), 1, (k // 2, 0), groups=c, bias=False)
        self.bn_strip = nn.BatchNorm2d(c)
        self.pw = nn.Conv2d(c, c, 1, 1, 0)
        self.fc2 = Conv(c, c, 1, 1, act=False)

    def forward(self, x):
        t = self.fc1(x)
        a = torch.nn.functional.silu(self.bn_sq(self.square(t)))
        a = self.bn_strip(self.v(self.h(a)))
        a = self.pw(a)
        return self.fc2(a * t), a


class CGLU(nn.Module):
    """1x1 扩增(2c) split 成 X,V；X 支 3x3 深度可分离+GELU；门控 DWConv(A)⊙V；
    content ⊙ gate；1x1 压缩 + Dropout。"""
    def __init__(self, c, dropout=0.1):
        super().__init__()
        self.expand = nn.Conv2d(c, 2 * c, 1, 1, 0)
        self.dw3 = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.pw3 = nn.Conv2d(c, c, 1, 1, 0)
        self.gate_dw = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.compress = nn.Conv2d(c, c, 1, 1, 0)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, a):
        xb, vb = self.expand(x).chunk(2, 1)
        content = torch.nn.functional.gelu(self.pw3(self.dw3(xb)))
        gate = self.gate_dw(a) * vb
        fgate = content * gate
        return self.drop(self.compress(fgate))


class StripCGLUBottleneck(nn.Module):
    """pre-norm 两个串联残差子块：Strip Block -> CGLU。"""
    def __init__(self, c, shortcut=True, k=7, dropout=0.1):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(c)
        self.strip = StripAttention(c, k)
        self.norm2 = nn.BatchNorm2d(c)
        self.cglu = CGLU(c, dropout)
        self.add = shortcut

    def forward(self, x):
        s, a = self.strip(self.norm1(x))
        y = x + s
        y = y + self.cglu(self.norm2(y), a)
        return y


class C2f_StripCGLU(C2f):
    """C2f 外壳不变，内部 Bottleneck 全部替换为 StripCGLUBottleneck。"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=7, dropout=0.1):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.ModuleList(
            StripCGLUBottleneck(c_, shortcut, k, dropout) for _ in range(n))


# ====================== 动态替换 Trainer ======================
def make_trainer(k=7, dropout=0.1):
    class _Trainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg, weights, verbose)
            for idx in [2, 4]:
                original = model.model[idx]
                c1 = original.cv1.conv.in_channels
                c2 = original.cv2.conv.out_channels
                n = len(original.m)
                shortcut = original.m[0].add
                e = original.c / c2
                new = C2f_StripCGLU(c1, c2, n=n, shortcut=shortcut, e=e,
                                    k=k, dropout=dropout)
                new.i, new.f = original.i, original.f
                model.model[idx] = new
                print(f"  替换 model[{idx}] -> C2f_StripCGLU c1={c1} c2={c2} n={n}")
            print(f"  替换后总参数量: {sum(p.numel() for p in model.parameters()):,}")
            return model
    return _Trainer


# ====================== COCO 评估 ======================
def get_sorted_img_list(img_dir):
    suffixes = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    return sorted([f for f in img_dir.iterdir() if f.suffix.lower() in suffixes])


def generate_coco_gt(img_list, label_dir):
    coco_gt = {"images": [], "annotations": [], "categories": []}
    coco_gt["categories"].append({"id": 0, "name": "ship", "supercategory": "object"})
    ann_id = 0
    for img_id, img_path in enumerate(img_list):
        with Image.open(img_path) as img:
            w, h = img.size
        coco_gt["images"].append({"id": img_id, "file_name": img_path.name,
                                  "width": w, "height": h})
        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        with open(label_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        for line in lines:
            parts = line.split()
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            bw *= w; bh *= h
            bx = cx * w - bw / 2
            by = cy * h - bh / 2
            coco_gt["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": cls_id,
                "bbox": [round(bx, 3), round(by, 3), round(bw, 3), round(bh, 3)],
                "area": round(bw * bh, 3), "iscrowd": 0})
            ann_id += 1
    return coco_gt


def generate_coco_predictions(img_list, model):
    predictions = []
    for img_id, img_path in enumerate(img_list):
        result = model.predict(source=str(img_path), imgsz=IMG_SIZE,
                               conf=CONF_THRES, iou=IOU_THRES, device=DEVICE,
                               verbose=False)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        for i in range(len(boxes)):
            x1, y1, x2, y2 = xyxy[i]
            predictions.append({
                "image_id": img_id, "category_id": int(clss[i]),
                "bbox": [round(float(x1), 3), round(float(y1), 3),
                         round(float(x2 - x1), 3), round(float(y2 - y1), 3)],
                "score": round(float(confs[i]), 5)})
    return predictions


def evaluate(exp_dir):
    result_dir = exp_dir / "val_coco_result"
    result_dir.mkdir(parents=True, exist_ok=True)
    img_list = get_sorted_img_list(VAL_IMG_DIR)
    print(f"  验证集图片数: {len(img_list)}；生成真值...")
    coco_gt_data = generate_coco_gt(img_list, VAL_LABEL_DIR)
    eval_model = YOLO(str(exp_dir / "weights" / "best.pt"))
    pred_data = generate_coco_predictions(img_list, eval_model)
    print(f"  预测框总数: {len(pred_data)}")
    gt_path = result_dir / "val_gt_coco.json"
    pred_path = result_dir / "val_predictions.json"
    gt_path.write_text(json.dumps(coco_gt_data, ensure_ascii=False, indent=2), encoding="utf-8")
    pred_path.write_text(json.dumps(pred_data, ensure_ascii=False, indent=2), encoding="utf-8")
    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(str(pred_path))
    ce = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ce.evaluate(); ce.accumulate(); ce.summarize()
    s = ce.stats
    return {"mAP@0.5": round(float(s[1]), 4), "mAP@0.5:0.95": round(float(s[0]), 4),
            "AP-small": round(float(s[3]), 4), "AP-medium": round(float(s[4]), 4),
            "AP-large": round(float(s[5]), 4)}


# ====================== 配对统计（无 scipy）======================
def betacf(a, b, x):
    MAXIT = 200; EPS = 3e-14; FPMIN = 1e-300
    qab = a + b; qap = a + 1; qam = a - 1; c = 1.; d = 1. - qab * x / qap
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
    if x <= 0.: return 0.
    if x >= 1.: return 1.
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return bt * betacf(a, b, x) / a
    return 1 - bt * betacf(b, a, 1 - x) / b


def load_baseline_metrics(seed):
    p = BASELINE_RUN_ROOT / f"HRSID_baseline_seed{seed}" / "val_coco_result" / "val_metrics.csv"
    with open(p, "r", encoding="utf-8-sig") as f:
        return next(csv.DictReader(f))


# ====================== 主流程 ======================
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in SEEDS:
        name = f"HRSID_stripcglu_seed{seed}"
        exp_dir = OUT / name
        metrics_csv = exp_dir / "val_coco_result" / "val_metrics.csv"
        if metrics_csv.exists():
            with open(metrics_csv, "r", encoding="utf-8-sig") as f:
                metrics = next(csv.DictReader(f))
            print(f"[跳过-已完成] {name}")
        else:
            print("\n" + "#" * 70)
            print(f"# 训练 {name} | k={K} seed={seed}")
            print("#" * 70)
            model = YOLO(INIT_WEIGHTS)
            trainer = make_trainer(K, DROPOUT)
            model.train(data=DATA_YAML, imgsz=IMG_SIZE, batch=BATCH, epochs=EPOCHS,
                        device=DEVICE, seed=seed, mosaic=MOSAIC, workers=WORKERS,
                        project=str(OUT), name=name, exist_ok=True,
                        trainer=trainer, verbose=True, amp=False)
            metrics = evaluate(exp_dir)
            metrics_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_csv, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f); w.writerow(METRIC_KEYS)
                w.writerow([metrics[k] for k in METRIC_KEYS])
            print(f"  已保存: {metrics_csv}")
        rec = {"seed": seed}; rec.update(metrics); runs.append(rec)

    # 1) 单次明细
    runs_path = OUT / "stripcglu_runs.csv"
    with open(runs_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["seed"] + METRIC_KEYS)
        w.writeheader(); w.writerows(runs)

    # 2) mean ± std
    mean_row = {"n": len(runs)}
    for k in METRIC_KEYS:
        v = np.array([float(r[k]) for r in runs])
        mean_row[f"{k}_mean"] = round(float(v.mean()), 4)
        mean_row[f"{k}_std"] = round(float(v.std(ddof=1)), 4)
    mean_fields = ["n"] + [f"{k}_{s}" for k in METRIC_KEYS for s in ("mean", "std")]
    mean_path = OUT / "stripcglu_mean_std.csv"
    with open(mean_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=mean_fields)
        w.writeheader(); w.writerow(mean_row)

    # 3) 与 baseline 配对 t 检验
    base_recs = [load_baseline_metrics(s) for s in SEEDS]
    n = len(SEEDS); df = n - 1
    ttest_rows = []
    print("\n" + "=" * 70)
    print("C2f_StripCGLU vs baseline（5 共同种子配对，差值=stripcglu-baseline）")
    print("=" * 70)
    for k in METRIC_KEYS:
        d = np.array([float(r[k]) for r in runs]) - np.array([float(b[k]) for b in base_recs])
        sd = d.std(ddof=1)
        t = d.mean() / (sd / math.sqrt(n))
        p = betai(df / 2, 0.5, df / (df + t * t))
        ttest_rows.append({"metric": k, "mean_diff": round(float(d.mean()), 4),
                           "diff_std": round(float(sd), 4), "t": round(float(t), 3),
                           "p": round(float(p), 4),
                           "diffs": json.dumps(np.round(d, 4).tolist())})
        print(f"  {k:<16} mean_diff={d.mean():+.4f}  t={t:+.3f}  p={p:.4f}")
    ttest_path = OUT / "stripcglu_vs_baseline_ttest.csv"
    with open(ttest_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "mean_diff", "diff_std", "t", "p", "diffs"])
        w.writeheader(); w.writerows(ttest_rows)

    print("\n" + "=" * 70)
    print("全部完成：")
    for pth in [runs_path, mean_path, ttest_path]:
        print(" ", pth)
    print("=" * 70)


if __name__ == "__main__":
    main()
