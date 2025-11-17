#!/usr/bin/env python3
"""Parse a SwanLab backup.swanlab file and plot multi-GPU comparisons.

Usage:
    python scripts/plot_swanlab_backup.py /path/to/backup.swanlab

Output:
    Creates PNG plots under the run's media/plots/ directory (created if needed).

This script extracts JSON objects from the binary backup and collects entries
with model_type == 'Scalar'. It groups per-metric (e.g. 'memory_used_mb',
'power_w', 'utilization', 'temperature', etc.) and draws one plot per metric
with one line per GPU (gpu_0, gpu_1, ...). Global metrics like gpu_avg/* are
also plotted separately.
"""
import sys
import os
import re
import json
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd


def iter_json_objects_from_bytes(data: bytes):
    """Yield JSON-decoded objects from a bytes blob by brace counting.

    We look for occurrences of b'{"model_type"' then scan forward keeping a
    counter of braces until the JSON object closes. This is robust for the
    structure in backup.swanlab (concatenated JSON objects).
    """
    start_token = b'{"model_type"'
    i = 0
    n = len(data)
    while True:
        j = data.find(start_token, i)
        if j == -1:
            return
        # find beginning '{'
        k = j
        if data[k] != 0x7b:  # '{'
            k = data.rfind(b'{', 0, j)
            if k == -1:
                i = j + 1
                continue
        brace = 0
        end = None
        for p in range(k, n):
            if data[p] == 0x7b:  # '{'
                brace += 1
            elif data[p] == 0x7d:  # '}'
                brace -= 1
                if brace == 0:
                    end = p + 1
                    break
        if end is None:
            return
        chunk = data[k:end]
        try:
            text = chunk.decode('utf-8')
        except Exception:
            # fallback: replace errors
            text = chunk.decode('utf-8', errors='replace')
        try:
            obj = json.loads(text)
            yield obj
        except Exception:
            # skip unparseable chunk
            pass
        i = end


def parse_backup(path):
    with open(path, 'rb') as f:
        data = f.read()
    for obj in iter_json_objects_from_bytes(data):
        yield obj


def collect_scalars(path):
    # key -> list of records (index, step, epoch, value, create_time)
    metrics = defaultdict(list)
    for obj in parse_backup(path):
        try:
            mtype = obj.get('model_type')
        except Exception:
            continue
        if mtype != 'Scalar':
            continue
        data = obj.get('data') or {}
        key = data.get('key')
        step = data.get('step')
        epoch = data.get('epoch')
        metric = data.get('metric') or {}
        index = metric.get('index')
        value = metric.get('data')
        create_time = metric.get('create_time')
        # normalize create_time to datetime if possible
        try:
            if create_time:
                create_time = datetime.fromisoformat(create_time.replace('Z','+00:00'))
        except Exception:
            create_time = None
        if key is None:
            continue
        metrics[key].append({'index': index, 'step': step, 'epoch': epoch, 'value': value, 'time': create_time})
    return metrics


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def plot_metric_group(metrics, metric_name, run_dir, ylabel=None, to_mb=False):
    """Plot all gpu_N/<metric_name> series on one figure.

    metrics: dict from key->list of records
    metric_name: e.g. 'memory_used_mb', 'power_w'
    """
    pattern = re.compile(rf'^gpu_(\d+)/{re.escape(metric_name)}$')
    series = {}
    for k, recs in metrics.items():
        m = pattern.match(k)
        if m:
            gpu = int(m.group(1))
            # sort by index/time
            df = pd.DataFrame(recs)
            if df.empty:
                continue
            if 'index' in df.columns and df['index'].notnull().any():
                df = df.sort_values('index')
                x = df['index']
            elif 'time' in df.columns and df['time'].notnull().any():
                df = df.sort_values('time')
                x = df['time']
            else:
                x = range(len(df))
            y = df['value'].astype(float)
            if to_mb and 'memory_used_gb' in metric_name:
                # if user requested conversion, skip (already GB)
                pass
            series[gpu] = (x.tolist(), y.tolist())
    if not series:
        return False
    plt.figure(figsize=(10, 6))
    for gpu in sorted(series.keys()):
        x, y = series[gpu]
        plt.plot(x, y, label=f'gpu_{gpu}')
    plt.xlabel('sample')
    plt.ylabel(ylabel or metric_name)
    plt.title(metric_name)
    plt.legend()
    plt.grid(True)
    out_dir = os.path.join(run_dir, 'media', 'plots')
    ensure_dir(out_dir)
    outpath = os.path.join(out_dir, f'{metric_name}.png')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    return True


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/plot_swanlab_backup.py /path/to/backup.swanlab')
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.isfile(path):
        print('backup file not found:', path)
        sys.exit(1)
    run_dir = os.path.dirname(path)
    print('Parsing', path)
    metrics = collect_scalars(path)
    if not metrics:
        print('No scalar metrics found in backup.')
        sys.exit(1)

    # list of metric names to plot (per-GPU)
    per_gpu_metrics = [
        'memory_used_mb',
        'power_w',
        'utilization',
        'temperature',
        'memory_util',
    ]

    plotted = []
    for m in per_gpu_metrics:
        ok = plot_metric_group(metrics, m, run_dir, ylabel=m)
        if ok:
            plotted.append(m)

    # also plot gpu_avg/* metrics (single line)
    avg_pattern = re.compile(r'^gpu_avg/(.+)$')
    for k in sorted(metrics.keys()):
        m = avg_pattern.match(k)
        if m:
            metric_name = 'gpu_avg_' + m.group(1)
            # create a simple plot for single series
            recs = metrics[k]
            df = pd.DataFrame(recs).sort_values('index')
            if df.empty:
                continue
            plt.figure(figsize=(8, 4))
            x = df['index'] if 'index' in df.columns else range(len(df))
            y = df['value'].astype(float)
            plt.plot(x, y, label=k)
            plt.xlabel('sample')
            plt.ylabel(m.group(1))
            plt.title(metric_name)
            plt.grid(True)
            out_dir = os.path.join(run_dir, 'media', 'plots')
            ensure_dir(out_dir)
            outpath = os.path.join(out_dir, f'{metric_name}.png')
            plt.tight_layout()
            plt.savefig(outpath)
            plt.close()
            plotted.append(metric_name)

    print('Plotted metrics:', plotted)
    print('Plots are saved under', os.path.join(run_dir, 'media', 'plots'))


if __name__ == '__main__':
    main()
