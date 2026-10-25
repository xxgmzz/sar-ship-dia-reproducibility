# -*- coding: utf-8 -*-
"""
SSDD n5 并行调度器（发布包版本）
================================
10 个任务（baseline / dia_l2_k3 × 5 种子），GPU 并发 4。
单进程 GPU 利用率仅约 20%，瓶颈在 workers=0 的 CPU 数据加载；
每进程显存约 1.6 GB，4 并发 < 8 GB。
任务按 seed 交错：每个 seed 的 baseline / dia 排在同一批，尽早凑成配对。
全部完成后自动调用 train_ssdd_n5.py --summary 汇总。

使用前：
1) 修改下方配置区的 PY（Python 解释器）与 OUT（输出根目录）；
2) SCRIPT / WORKDIR 已自动指向同目录的 train_ssdd_n5.py，通常无需改动。
"""
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ====================== 配置区（按需修改）======================
PY = r"D:\miniconda3\envs\det\python.exe"          # 你的 Python 解释器
OUT = Path(r"D:\SAR_Det_Research\ssdd_multiseed_n5")  # 输出根目录
# 以下两项自动定位同目录脚本，一般无需修改
SCRIPT = str(Path(__file__).resolve().parent / "train_ssdd_n5.py")
WORKDIR = str(Path(__file__).resolve().parent)

LOGS = OUT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 2024, 7, 2025]
VARIANTS = ["baseline", "dia_l2_k3"]
PARALLEL = 4

# 按 seed 交错：每个 seed 的 baseline + dia 相邻
TASKS = []
for s in SEEDS:
    for v in VARIANTS:
        TASKS.append((v, s))


def run_task(variant, seed):
    log_path = LOGS / f"{variant}_seed{seed}.log"
    err_path = LOGS / f"{variant}_seed{seed}.err"
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as lf, open(err_path, "w", encoding="utf-8") as ef:
        proc = subprocess.run(
            [PY, SCRIPT, "--variant", variant, "--seed", str(seed)],
            stdout=lf, stderr=ef, cwd=WORKDIR,
        )
    return variant, seed, proc.returncode, time.time() - t0


def main():
    print(f"[调度器] 共 {len(TASKS)} 任务，并发 {PARALLEL}", flush=True)
    print(f"[调度器] 训练脚本: {SCRIPT}", flush=True)
    start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(run_task, v, s): (v, s) for v, s in TASKS}
        for fut in as_completed(futs):
            v, s, rc, dt = fut.result()
            results.append((v, s, rc))
            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            print(f"[{time.strftime('%H:%M:%S')}] {v} seed{s} {status} "
                  f"用时 {dt/60:.1f} 分钟，累计 {(time.time()-start)/60:.1f} 分钟", flush=True)

    print("\n[调度器] 全部任务结束，开始汇总...", flush=True)
    sumlog = open(LOGS / "summary.log", "w", encoding="utf-8")
    sumerr = open(LOGS / "summary.err", "w", encoding="utf-8")
    p = subprocess.run([PY, SCRIPT, "--summary"], stdout=sumlog, stderr=sumerr, cwd=WORKDIR)
    sumlog.close(); sumerr.close()
    print(f"[调度器] 汇总退出码 {p.returncode}，总耗时 {(time.time()-start)/60:.1f} 分钟", flush=True)
    fail = [r for r in results if r[2] != 0]
    if fail:
        print(f"[调度器] 失败任务：{fail}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
