# -*- coding: utf-8 -*-
"""
SSDD 多种子配对实验（n=5）—— 最终配置 L2-only k=3
======================================================
目的：把 SSDD 上"无变体优于基线"的结论从单次运行(seed42, k=5 早期消融)升级为
最终配置(L2-only, k=3, 与 HRSID/HRSC 完全一致)的配对统计结论。
两配置：baseline(layers=[]) 与 dia_l2_k3(layers=[2], k=3)
各 5 个配对种子 [42,123,2024,7,2025]，同数据划分、同超参数、同初始权重。
超参数与 HRSID/HRSC 主线完全一致：img640,batch8,epoch200,mosaic0,workers0。
断点续跑：已有 val_metrics.csv 自动跳过。
"""
import csv
import json
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
SEEDS = [42, 123, 2024, 7, 2025]
IMG_SIZE = 640
BATCH_SIZE = 8
EPOCHS = 200
DEVICE = 0
MOSAIC = 0
WORKERS = 0
CONF_THRES = 0.001
IOU_THRES = 0.6

INIT_WEIGHTS = r"D:\SAR_Det_Research\exp00_baseline_yolov8n_ssdd\yolov8n.pt"
OUT_ROOT = Path(r"D:\SAR_Det_Research\ssdd_multiseed_n5")

DATA = {
    "data_yaml": r"D:\SAR_Det_Research\exp00_baseline_yolov8n_ssdd\ssdd.yaml",
    "val_img": Path(r"D:\SAR_Det_Research\exp00_baseline_yolov8n_ssdd\sar_ssdd\images\val"),
    "val_label": Path(r"D:\SAR_Det_Research\exp00_baseline_yolov8n_ssdd\sar_ssdd\labels\val"),
    "variants": {
        "baseline":  {"layers": [],  "k": None},
        "dia_l2_k3": {"layers": [2], "k": 3},
    },
}
METRIC_KEYS = ["mAP@0.5", "mAP@0.5:0.95", "AP-small", "AP-medium", "AP-large"]

# ====================== 自定义模块（与 HRSID 主线完全一致，支持 k 参数）======================
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


def evaluate(exp_dir):
    result_dir = exp_dir / "val_coco_result"
    result_dir.mkdir(parents=True, exist_ok=True)
    img_list = get_sorted_img_list(DATA["val_img"])
    print(f"  验证集图片数: {len(img_list)}；生成真值...")
    coco_gt_data = generate_coco_gt(img_list, DATA["val_label"])
    best_weight = exp_dir / "weights" / "best.pt"
    eval_model = YOLO(str(best_weight))
    pred_data = generate_coco_predictions(img_list, eval_model)
    print(f"  预测框总数: {len(pred_data)}")
    gt_path = result_dir / "val_gt_coco.json"
    pred_path = result_dir / "val_predictions.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(coco_gt_data, f, ensure_ascii=False)
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(pred_data, f, ensure_ascii=False)
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


def run_one(variant_name, seed):
    vcfg = DATA["variants"][variant_name]
    name = f"SSDD_{variant_name}_seed{seed}"
    exp_dir = OUT_ROOT / name
    metrics_csv = exp_dir / "val_coco_result" / "val_metrics.csv"
    record = {"dataset": "SSDD", "variant": variant_name, "seed": seed}
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
        data=DATA["data_yaml"], imgsz=IMG_SIZE, batch=BATCH_SIZE, epochs=EPOCHS,
        device=DEVICE, seed=seed, mosaic=MOSAIC, workers=WORKERS,
        project=str(OUT_ROOT), name=name, exist_ok=True,
        trainer=trainer_cls, verbose=True,
    )
    metrics = evaluate(exp_dir)
    record.update(metrics)
    with open(metrics_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(METRIC_KEYS)
        w.writerow([metrics[k] for k in METRIC_KEYS])
    print(f"  已保存指标: {metrics_csv}")
    return record


def summarize(all_records):
    import numpy as np
    runs_csv = OUT_ROOT / "ssdd_multiseed_runs.csv"
    fields = ["dataset", "variant", "seed"] + METRIC_KEYS
    with open(runs_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_records:
            w.writerow(r)
    groups = {}
    for r in all_records:
        groups.setdefault(r["variant"], []).append(r)
    mean_fields = ["variant", "n"]
    for k in METRIC_KEYS:
        mean_fields += [f"{k}_mean", f"{k}_std"]
    mean_rows = []
    for var, recs in sorted(groups.items()):
        numeric = [rec for rec in recs if rec.get(METRIC_KEYS[0]) not in (None, "")]
        row = {"variant": var, "n": len(numeric)}
        for k in METRIC_KEYS:
            vals = np.array([float(rec[k]) for rec in numeric])
            row[f"{k}_mean"] = round(float(vals.mean()), 4) if len(vals) else ""
            row[f"{k}_std"] = round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else ""
        mean_rows.append(row)
    mean_csv = OUT_ROOT / "ssdd_mean_std.csv"
    with open(mean_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=mean_fields)
        w.writeheader()
        w.writerows(mean_rows)
    print("\n" + "=" * 70)
    for row in mean_rows:
        print(f"SSDD {row['variant']} (n={row['n']})")
        for k in METRIC_KEYS:
            print(f"   {k:<16} {row[f'{k}_mean']} ± {row[f'{k}_std']}")
    print("=" * 70)


def collect_records():
    """从各任务 val_metrics.csv 收集结果（含已完成的）。"""
    all_records = []
    for var_name in DATA["variants"]:
        for seed in SEEDS:
            name = f"SSDD_{var_name}_seed{seed}"
            csv_path = OUT_ROOT / name / "val_coco_result" / "val_metrics.csv"
            if not csv_path.exists():
                continue
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                continue
            row = rows[0]
            rec = {"dataset": "SSDD", "variant": var_name, "seed": seed}
            for k in METRIC_KEYS:
                rec[k] = row.get(k)
            all_records.append(rec)
    return all_records


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default=None, help="只跑指定变体：baseline / dia_l2_k3")
    parser.add_argument("--seed", type=int, default=None, help="只跑指定种子")
    parser.add_argument("--summary", action="store_true", help="只汇总已有结果，不训练")
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.summary:
        summarize(collect_records())
        return
    if args.variant is not None and args.seed is not None:
        # 单任务模式（供并行调度器 subprocess 调用）
        run_one(args.variant, args.seed)
        return
    all_records = []
    for var_name in DATA["variants"]:
        for seed in SEEDS:
            try:
                all_records.append(run_one(var_name, seed))
            except Exception as e:
                import traceback
                print(f"[失败] {var_name}/seed{seed}: {e}")
                traceback.print_exc()
    summarize(all_records)


if __name__ == "__main__":
    main()
