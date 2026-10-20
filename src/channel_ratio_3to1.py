# -*- coding: utf-8 -*-
"""
HRSID 通道比补充实验（选项 B）
=====================================================
配置：DIA-Conv L2-only、k=3、通道比 3:1（各向同性 3/4、各向异性 1/4）、seed=42 单次。
对照：1:1 通道比 seed=42（已在 multiseed_experiments，n=8 已跑完）。
输出根：D:\\SAR_Det_Research\\channel_ratio_experiments
  - HRSID_dia_l2_k3_r3to1_seed42\\weights\\best.pt
  - HRSID_dia_l2_k3_r3to1_seed42\\val_coco_result\\val_metrics.csv
  - channel_ratio_compare.csv（1:1 vs 3:1，seed42 并排）
与成功脚本一致：不依赖模型结构 yaml，从 yolov8n.pt 加载、自定义 trainer 动态替换、COCO 评估。
建议脚本名：run_hrsid_channel_ratio_3to1.py
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

# ====================== 配置 ======================
SEED = 42
IMG_SIZE = 640
BATCH_SIZE = 8
EPOCHS = 200
DEVICE = 0
MOSAIC = 0
WORKERS = 0

CONF_THRES = 0.001
IOU_THRES = 0.6

ORIGIN_RATIO = 0.75          # 3:1：各向同性占 3/4，各向异性占 1/4
REPLACE_LAYERS = [2]
KERNEL_SIZE = 3

INIT_WEIGHTS = r"D:\SAR_Det_Research\exp00_baseline_yolov8n_ssdd\yolov8n.pt"
DATA_YAML = r"D:\SAR_Det_Research\hrsid_experiments\configs\hrsid.yaml"
VAL_IMG = Path(r"D:\SAR_Det_Research\hrsid_experiments\data\images\val")
VAL_LABEL = Path(r"D:\SAR_Det_Research\hrsid_experiments\data\labels\val")

OUT_ROOT = Path(r"D:\SAR_Det_Research\channel_ratio_experiments")
EXP_NAME = "HRSID_dia_l2_k3_r3to1_seed42"

# 1:1 seed42 参照（n=8 已完成）
R1TO1_CSV = Path(r"D:\SAR_Det_Research\multiseed_experiments\HRSID_dia_l2_k3_seed42\val_coco_result\val_metrics.csv")

METRIC_KEYS = ["mAP@0.5", "mAP@0.5:0.95", "AP-small", "AP-medium", "AP-large"]


# ====================== 自定义模块（支持 origin_ratio）======================
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
    def __init__(self, c, shortcut=True, k=3, origin_ratio=0.5):
        super().__init__()
        self.shortcut = shortcut
        c_origin = int(round(c * origin_ratio))
        c_strip = c - c_origin
        assert c_origin >= 1 and c_strip >= 1, f"bad channel split: {c} -> {c_origin}/{c_strip}"
        self.branch_origin = Conv(c, c_origin, 3, 1)
        self.branch_strip = StripConv(c, c_strip, k=k)

    def forward(self, x):
        x_origin = self.branch_origin(x)
        x_strip = self.branch_strip(x)
        out = torch.cat([x_origin, x_strip], dim=1)
        return x + out if self.shortcut else out


class C2f_DualBranch(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3, origin_ratio=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            DualBranchBottleneck(self.c, shortcut=shortcut, k=k, origin_ratio=origin_ratio)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def make_trainer(replace_layers, kernel_size, origin_ratio):
    class _Trainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg, weights, verbose)
            if replace_layers:
                print("\n" + "=" * 70)
                print(f"动态替换 C2f：layers={replace_layers}, k={kernel_size}, origin_ratio={origin_ratio}")
                for idx in replace_layers:
                    original = model.model[idx]
                    c1 = original.cv1.conv.in_channels
                    c2 = original.cv2.conv.out_channels
                    n = len(original.m)
                    shortcut = original.m[0].add
                    e = original.c / c2
                    new_layer = C2f_DualBranch(c1, c2, n=n, shortcut=shortcut, e=e,
                                               k=kernel_size, origin_ratio=origin_ratio)
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
    img_list = get_sorted_img_list(VAL_IMG)
    print(f"  验证集图片数: {len(img_list)}；生成真值...")
    coco_gt_data = generate_coco_gt(img_list, VAL_LABEL)

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


def read_metrics_csv(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return rows[0]


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    exp_dir = OUT_ROOT / EXP_NAME
    metrics_csv = exp_dir / "val_coco_result" / "val_metrics.csv"

    # 断点续跑
    if metrics_csv.exists():
        print(f"[跳过-已完成] {EXP_NAME}")
        m3to1 = read_metrics_csv(metrics_csv)
    else:
        print("\n" + "#" * 70)
        print(f"# 训练 {EXP_NAME} | layers={REPLACE_LAYERS} k={KERNEL_SIZE} "
              f"ratio={ORIGIN_RATIO} seed={SEED}")
        print("#" * 70)
        model = YOLO(INIT_WEIGHTS)
        trainer_cls = make_trainer(REPLACE_LAYERS, KERNEL_SIZE, ORIGIN_RATIO)
        model.train(
            data=DATA_YAML, imgsz=IMG_SIZE, batch=BATCH_SIZE, epochs=EPOCHS,
            device=DEVICE, seed=SEED, mosaic=MOSAIC, workers=WORKERS,
            project=str(OUT_ROOT), name=EXP_NAME, exist_ok=True,
            trainer=trainer_cls, verbose=True,
        )
        metrics = evaluate(exp_dir)
        with open(metrics_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(METRIC_KEYS)
            w.writerow([metrics[k] for k in METRIC_KEYS])
        print(f"  已保存指标: {metrics_csv}")
        m3to1 = metrics

    # 并排对照（1:1 seed42 vs 3:1 seed42）
    m1to1 = read_metrics_csv(R1TO1_CSV) if R1TO1_CSV.exists() else {}
    compare_csv = OUT_ROOT / "channel_ratio_compare.csv"
    with open(compare_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["channel_ratio"] + METRIC_KEYS)
        w.writerow(["1:1 (seed42)"] + [m1to1.get(k, "") for k in METRIC_KEYS])
        w.writerow(["3:1 (seed42)"] + [m3to1.get(k, "") for k in METRIC_KEYS])

    print("\n" + "=" * 70)
    print("通道比对照（seed42 单次）")
    print("=" * 70)
    for k in METRIC_KEYS:
        print(f"  {k:<16} 1:1={m1to1.get(k,'')}   3:1={m3to1.get(k,'')}")
    print("=" * 70)
    print(f"对照表: {compare_csv}")


if __name__ == "__main__":
    main()
