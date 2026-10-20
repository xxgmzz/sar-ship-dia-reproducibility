# -*- coding: utf-8 -*-
"""
多种子重复实验 n=5 -> n=8 续跑脚本（HRSID 主线）
=====================================================
在 run_multiseed_experiments.py 已完成的 5 种子（42/123/2024/7/2025）基础上，
新增 3 个种子（11/88/314），共 2 配置 × 8 种子。
样本量 n=8 依据阶段 A bootstrap/MDE 功效分析确定：
对当前唯一边缘正向的 mAP@0.5:0.95 效应（n=5 时 p=0.075），n=8 即可达到
t≈3.0（p≈0.02）的定案能力；AP-large 均值在 n=8 下预计仍不显著（诚实负结果）。
断点续跑：已完成的 5 种子（含 val_metrics.csv）自动跳过，本次只训练新增 3×2=6 次。
训练结束后自动用全部 8 种子重写：
  multiseed_runs.csv（全部单次）+ multiseed_mean_std.csv（mean±std）
其余设置与原脚本完全一致：不依赖模型结构 yaml、从 yolov8n.pt 加载、动态替换、COCO 评估。
建议脚本名：run_multiseed_to_n8.py
"""
import csv
import json
import shutil
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import Conv
from ultralytics.models.yolo.detect import DetectionTrainer
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ====================== 全局配置 ======================
# 前 5 个已完成（自动跳过），后 3 个为本次新增
SEEDS = [42, 123, 2024, 7, 2025, 11, 88, 314]
IMG_SIZE = 640
BATCH_SIZE = 8
EPOCHS = 200
DEVICE = 0
MOSAIC = 0
WORKERS = 0

CONF_THRES = 0.001
IOU_THRES = 0.6
CLASS_NAMES = ["ship"]

# 统一初始权重（标准 yolov8n.pt，SSDD 实验目录下已存在）
INIT_WEIGHTS = r"D:\SAR_Det_Research\exp00_baseline_yolov8n_ssdd\yolov8n.pt"

# 统一输出根目录（与原脚本一致，新增种子目录与原有并列）
OUT_ROOT = Path(r"D:\SAR_Det_Research\multiseed_experiments")

RUN_DATASETS = ["HRSID"]

# 数据集定义
DATASETS = {
    "HRSID": {
        "data_yaml": r"D:\SAR_Det_Research\hrsid_experiments\configs\hrsid.yaml",
        "val_img": Path(r"D:\SAR_Det_Research\hrsid_experiments\data\images\val"),
        "val_label": Path(r"D:\SAR_Det_Research\hrsid_experiments\data\labels\val"),
        "variants": {
            "baseline":    {"layers": [],       "k": None},
            "dia_l2_k3":   {"layers": [2],      "k": 3},
        },
    },
    "SSDD": {
        "data_yaml": r"D:\SAR_Det_Research\exp00_baseline_yolov8n_ssdd\ssdd.yaml",
        "val_img": Path(r"D:\SAR_Det_Research\exp00_baseline_yolov8n_ssdd\sar_ssdd\images\val"),
        "val_label": Path(r"D:\SAR_Det_Research\exp00_baseline_yolov8n_ssdd\sar_ssdd\labels\val"),
        "variants": {
            "baseline":      {"layers": [],     "k": None},
            "dia_l2l4_k5":   {"layers": [2, 4], "k": 5},
        },
    },
}

METRIC_KEYS = ["mAP@0.5", "mAP@0.5:0.95", "AP-small", "AP-medium", "AP-large"]

# ====================== 自定义模块（与已跑实验完全一致）======================
class StripConv(nn.Module):
    def __init__(self, c1, c2, k=5, stride=1, act=True):
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
    def __init__(self, c, shortcut=True, k=5):
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
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            DualBranchBottleneck(self.c, shortcut=shortcut, k=k)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def make_trainer(replace_layers, kernel_size):
    """生成指定 替换层/卷积核 的 Trainer；replace_layers 为空时等价于基线"""
    class _Trainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg, weights, verbose)
            if replace_layers:
                print("\n" + "=" * 70)
                print(f"动态替换 C2f：layers={replace_layers}, k={kernel_size}")
                for idx in replace_layers:
                    original = model.model[idx]
                    c1 = original.cv1.conv.in_channels
                    c2 = original.cv2.conv.out_channels
                    n = len(original.m)
                    shortcut = original.m[0].add
                    e = original.c / c2
                    new_layer = C2f_DualBranch(c1, c2, n=n, shortcut=shortcut, e=e, k=kernel_size)
                    new_layer.f = original.f
                    new_layer.i = original.i
                    model.model[idx] = new_layer
                    print(f"  第{idx}层替换完成：C2f -> C2f_DualBranch，通道{c1}->{c2}，n={n}，k={kernel_size}")
                print(f"  替换后总参数量：{sum(p.numel() for p in model.parameters()):,}")
                print("=" * 70 + "\n")
            return model
    return _Trainer


# ====================== COCO 评估工具 ======================
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
        coco_gt["images"].append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})
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
                "area": round(bw * bh, 3), "iscrowd": 0
            })
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
                "score": round(float(confs[i]), 5)
            })
    return predictions


def evaluate(exp_dir, ds_cfg):
    """在指定实验目录内做 COCO 评估，返回指标 dict"""
    result_dir = exp_dir / "val_coco_result"
    result_dir.mkdir(parents=True, exist_ok=True)
    img_list = get_sorted_img_list(ds_cfg["val_img"])
    print(f"  验证集图片数: {len(img_list)}；生成真值...")
    coco_gt_data = generate_coco_gt(img_list, ds_cfg["val_label"])

    best_weight = exp_dir / "weights" / "best.pt"
    eval_model = YOLO(str(best_weight))
    pred_data = generate_coco_predictions(img_list, eval_model)
    print(f"  预测框总数: {len(pred_data)}")

    gt_path = result_dir / "val_gt_coco.json"
    pred_path = result_dir / "val_predictions.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(coco_gt_data, f, ensure_ascii=False, indent=2)
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(pred_data, f, ensure_ascii=False, indent=2)

    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(str(pred_path))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    stats = coco_eval.stats
    return {
        "mAP@0.5": round(float(stats[1]), 4),
        "mAP@0.5:0.95": round(float(stats[0]), 4),
        "AP-small": round(float(stats[3]), 4),
        "AP-medium": round(float(stats[4]), 4),
        "AP-large": round(float(stats[5]), 4),
    }


# ====================== 主流程 ======================
def run_one(dataset_name, variant_name, seed):
    ds = DATASETS[dataset_name]
    vcfg = ds["variants"][variant_name]
    name = f"{dataset_name}_{variant_name}_seed{seed}"
    exp_dir = OUT_ROOT / name
    metrics_csv = exp_dir / "val_coco_result" / "val_metrics.csv"

    record = {"dataset": dataset_name, "variant": variant_name, "seed": seed}

    # 断点续跑：已有指标则直接读取
    if metrics_csv.exists():
        with open(metrics_csv, "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        row = rows[0]
        for k in METRIC_KEYS:
            record[k] = row.get(k)
        print(f"[跳过-已完成] {name}")
        return record

    print("\n" + "#" * 70)
    print(f"# 训练 {name} | layers={vcfg['layers']} k={vcfg['k']} seed={seed}")
    print("#" * 70)

    model = YOLO(INIT_WEIGHTS)
    trainer_cls = make_trainer(vcfg["layers"], vcfg["k"])
    model.train(
        data=ds["data_yaml"], imgsz=IMG_SIZE, batch=BATCH_SIZE, epochs=EPOCHS,
        device=DEVICE, seed=seed, mosaic=MOSAIC, workers=WORKERS,
        project=str(OUT_ROOT), name=name, exist_ok=True,
        trainer=trainer_cls, verbose=True,
    )

    metrics = evaluate(exp_dir, ds)
    record.update(metrics)

    # 保存单次指标 CSV
    with open(metrics_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(METRIC_KEYS)
        w.writerow([metrics[k] for k in METRIC_KEYS])
    print(f"  已保存指标: {metrics_csv}")
    return record


def summarize(all_records):
    # 1) 全部单次
    runs_csv = OUT_ROOT / "multiseed_runs.csv"
    fields = ["dataset", "variant", "seed"] + METRIC_KEYS
    with open(runs_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_records:
            w.writerow(r)

    # 2) mean ± std
    import numpy as np
    groups = {}
    for r in all_records:
        key = (r["dataset"], r["variant"])
        groups.setdefault(key, []).append(r)

    mean_fields = ["dataset", "variant", "n"]
    for k in METRIC_KEYS:
        mean_fields += [f"{k}_mean", f"{k}_std"]
    mean_rows = []
    for (ds, var), recs in sorted(groups.items()):
        numeric = [rec for rec in recs if rec.get(METRIC_KEYS[0]) not in (None, "")]
        row = {"dataset": ds, "variant": var, "n": len(numeric)}
        for k in METRIC_KEYS:
            vals = np.array([float(rec[k]) for rec in numeric])
            row[f"{k}_mean"] = round(float(vals.mean()), 4) if len(vals) else ""
            row[f"{k}_std"] = round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else ""
        mean_rows.append(row)

    mean_csv = OUT_ROOT / "multiseed_mean_std.csv"
    with open(mean_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=mean_fields)
        w.writeheader()
        w.writerows(mean_rows)

    print("\n" + "=" * 70)
    print("多种子结果 mean ± std")
    print("=" * 70)
    for row in mean_rows:
        print(f"{row['dataset']} {row['variant']} (n={row['n']})")
        for k in METRIC_KEYS:
            m, s = row[f"{k}_mean"], row[f"{k}_std"]
            print(f"   {k:<16} {m} ± {s}")
    print("=" * 70)
    print(f"单次明细: {runs_csv}")
    print(f"汇总表:   {mean_csv}")


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_records = []
    for ds_name in RUN_DATASETS:
        ds = DATASETS[ds_name]
        for var_name in ds["variants"]:
            for seed in SEEDS:
                try:
                    rec = run_one(ds_name, var_name, seed)
                    all_records.append(rec)
                except Exception as e:
                    import traceback
                    print(f"[失败] {ds_name}/{var_name}/seed{seed}: {e}")
                    traceback.print_exc()
    summarize(all_records)


if __name__ == "__main__":
    main()
