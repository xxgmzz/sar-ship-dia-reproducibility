# -*- coding: utf-8 -*-
"""
跨域泛化评估（n=8）：HRSID 训练 -> SSDD 测试（不训练，仅推理）
=====================================================================
模型（均在 HRSID 上训练，8 种子 42/123/2024/7/2025/11/88/314）：
  - baseline : multiseed_experiments/HRSID_baseline_seed{s}/weights/best.pt
  - DIA      : multiseed_experiments/HRSID_dia_l2_k3_seed{s}/weights/best.pt
测试集：SSDD 验证集（348 图）
输出（D:\\SAR_Det_Research\\crossdomain_hrsid_to_ssdd）：
  - crossdomain_runs.csv      两组×8种子 单次指标
  - crossdomain_mean_std.csv  两组 mean±std
  - crossdomain_ttest.csv     baseline vs DIA 配对 t（含 p、逐种子差值）
注：原 n=5（含 ALK-DIA）结果已备份至 crossdomain_hrsid_to_ssdd_n5_backup。
建议脚本名：run_crossdomain_eval_n8.py
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
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ====================== 配置 ======================
SEEDS = [42, 123, 2024, 7, 2025, 11, 88, 314]
IMG_SIZE = 640
DEVICE = 0
CONF_THRES = 0.001
IOU_THRES = 0.6

SSDD_IMG = Path(r"D:\SAR_Det_Research\exp00_baseline_yolov8n_ssdd\sar_ssdd\images\val")
SSDD_LBL = Path(r"D:\SAR_Det_Research\exp00_baseline_yolov8n_ssdd\sar_ssdd\labels\val")

MULTI = Path(r"D:\SAR_Det_Research\multiseed_experiments")
OUT = Path(r"D:\SAR_Det_Research\crossdomain_hrsid_to_ssdd")

GROUPS = {
    "baseline": {s: MULTI / f"HRSID_baseline_seed{s}" / "weights" / "best.pt" for s in SEEDS},
    "dia":      {s: MULTI / f"HRSID_dia_l2_k3_seed{s}" / "weights" / "best.pt" for s in SEEDS},
}
METRIC_KEYS = ["mAP@0.5", "mAP@0.5:0.95", "AP-small", "AP-medium", "AP-large"]

# ====================== 自定义模块（供 pickle 反序列化，须与训练脚本逐字一致）======================
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
            DualBranchBottleneck(self.c, shortcut=shortcut, k=k) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ====================== SSDD COCO 真值（只生成一次）======================
def get_img_list():
    suffixes = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    return sorted([f for f in SSDD_IMG.iterdir() if f.suffix.lower() in suffixes])


def build_gt(img_list):
    gt = {"images": [], "annotations": [],
          "categories": [{"id": 0, "name": "ship", "supercategory": "object"}]}
    ann = 0
    for img_id, p in enumerate(img_list):
        with Image.open(p) as im:
            w, h = im.size
        gt["images"].append({"id": img_id, "file_name": p.name, "width": w, "height": h})
        lp = SSDD_LBL / (p.stem + ".txt")
        if not lp.exists():
            continue
        for line in lp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            q = line.split(); cid = int(q[0]); cx, cy, bw, bh = map(float, q[1:5])
            bw *= w; bh *= h; bx = cx*w - bw/2; by = cy*h - bh/2
            gt["annotations"].append({
                "id": ann, "image_id": img_id, "category_id": cid,
                "bbox": [round(bx, 3), round(by, 3), round(bw, 3), round(bh, 3)],
                "area": round(bw*bh, 3), "iscrowd": 0})
            ann += 1
    return gt

# ====================== 单模型评估 ======================
def predict_coco(img_list, model):
    preds = []
    for img_id, p in enumerate(img_list):
        r = model.predict(source=str(p), imgsz=IMG_SIZE, conf=CONF_THRES,
                         iou=IOU_THRES, device=DEVICE, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy(); confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = xyxy[i]
            preds.append({"image_id": img_id, "category_id": int(clss[i]),
                         "bbox": [round(float(x1), 3), round(float(y1), 3),
                                  round(float(x2-x1), 3), round(float(y2-y1), 3)],
                         "score": round(float(confs[i]), 5)})
    return preds


def eval_weight(gt_path, img_list, weight):
    model = YOLO(str(weight))
    preds = predict_coco(img_list, model)
    pred_path = OUT / "_tmp_pred.json"
    pred_path.write_text(json.dumps(preds, ensure_ascii=False), encoding="utf-8")
    coco_gt = COCO(str(gt_path)); coco_dt = coco_gt.loadRes(str(pred_path))
    ce = COCOeval(coco_gt, coco_dt, "bbox"); ce.evaluate(); ce.accumulate(); ce.summarize()
    s = ce.stats
    return {"mAP@0.5": round(float(s[1]), 4), "mAP@0.5:0.95": round(float(s[0]), 4),
            "AP-small": round(float(s[3]), 4), "AP-medium": round(float(s[4]), 4),
            "AP-large": round(float(s[5]), 4)}

# ====================== 配对 t（自包含）======================
def betacf(a, b, x):
    MAXIT = 200; EPS = 3e-14; FPMIN = 1e-300
    qab = a+b; qap = a+1; qam = a-1; c = 1.; d = 1.-qab*x/qap
    if abs(d) < FPMIN: d = FPMIN
    d = 1./d; h = d
    for m in range(1, MAXIT+1):
        m2 = 2*m
        aa = m*(b-m)*x/((qam+m2)*(a+m2)); d = 1.+aa*d
        if abs(d) < FPMIN: d = FPMIN
        c = 1.+aa/c
        if abs(c) < FPMIN: c = FPMIN
        d = 1./d; h *= d*c
        aa = -(a+m)*(qab+m)*x/((a+m2)*(qap+m2)); d = 1.+aa*d
        if abs(d) < FPMIN: d = FPMIN
        c = 1.+aa/c
        if abs(c) < FPMIN: c = FPMIN
        d = 1./d; de = d*c
        if abs(de-1.) < EPS: break
        h *= de
    return h


def betai(a, b, x):
    if x <= 0.: return 0.
    if x >= 1.: return 1.
    bt = math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log(1-x))
    if x < (a+1)/(a+b+2): return bt*betacf(a, b, x)/a
    return 1-bt*betacf(b, a, 1-x)/b


def paired_ttest(base, other):
    n = len(base); df = n-1; out = {}
    for k in METRIC_KEYS:
        d = np.array([float(x[k]) for x in other])-np.array([float(x[k]) for x in base])
        t = d.mean()/(d.std(ddof=1)/math.sqrt(n))
        p = betai(df/2, 0.5, df/(df+t*t))
        out[k] = {"mean_diff": round(float(d.mean()), 4), "t": round(float(t), 3),
                  "p": round(float(p), 4), "diffs": np.round(d, 4).tolist()}
    return out

# ====================== 主流程 ======================
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    img_list = get_img_list()
    print(f"SSDD 验证集图片数: {len(img_list)}；生成真值...")
    gt = build_gt(img_list)
    gt_path = OUT / "ssdd_gt_coco.json"
    gt_path.write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"真值已保存: {gt_path}")

    records = {g: [] for g in GROUPS}
    for g, wdict in GROUPS.items():
        for s in SEEDS:
            print("\n" + "#" * 60 + f"\n# 评估 {g} seed{s}（HRSID训练 -> SSDD测试）\n" + "#"*60)
            assert wdict[s].exists(), f"缺少权重: {wdict[s]}"
            rec = {"group": g, "seed": s}; rec.update(eval_weight(gt_path, img_list, wdict[s]))
            records[g].append(rec)

    # 单次明细
    runs_csv = OUT / "crossdomain_runs.csv"
    with open(runs_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, ["group", "seed"]+METRIC_KEYS); w.writeheader()
        for g in GROUPS:
            for r in records[g]:
                w.writerow(r)

    # mean±std
    def ms(rs):
        o = {}
        for k in METRIC_KEYS:
            v = np.array([float(x[k]) for x in rs]); o[k] = (round(v.mean(), 4), round(v.std(ddof=1), 4))
        return o
    mstat = {g: ms(records[g]) for g in GROUPS}
    ms_csv = OUT / "crossdomain_mean_std.csv"
    with open(ms_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["group", "n"]+[x for k in METRIC_KEYS for x in (k+"_mean", k+"_std")])
        for g in GROUPS:
            w.writerow([g, len(SEEDS)]+[x for k in METRIC_KEYS for x in mstat[g][k]])

    # 配对 t：baseline vs dia
    tt = {"dia_vs_baseline": paired_ttest(records["baseline"], records["dia"])}
    tt_csv = OUT / "crossdomain_ttest.csv"
    with open(tt_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["comparison", "metric", "mean_diff", "t", "p_two_sided", "per_seed_diffs"])
        for cmp_, d in tt.items():
            for k in METRIC_KEYS:
                w.writerow([cmp_, k, d[k]["mean_diff"], d[k]["t"], d[k]["p"], d[k]["diffs"]])

    (OUT / "_tmp_pred.json").unlink(missing_ok=True)

    print("\n"+"="*90)
    print(f"跨域泛化：HRSID 训练 -> SSDD 测试（n={len(SEEDS)}）")
    print("="*90)
    print(f"{'metric':<16}{'baseline':>16}{'DIA':>16}{'p(DIA)':>9}")
    for k in METRIC_KEYS:
        b = mstat["baseline"][k]; d = mstat["dia"][k]; pd_ = tt["dia_vs_baseline"][k]["p"]
        print(f"{k:<16}{f'{b[0]}±{b[1]}':>16}{f'{d[0]}±{d[1]}':>16}{pd_:>9}")
    print("="*90)
    print(f"明细: {runs_csv}\nmean±std: {ms_csv}\nt检验: {tt_csv}")


if __name__ == "__main__":
    main()
