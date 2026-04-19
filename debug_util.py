import csv
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


class DEBUG:
    enabled = False
    log_path = None
    context = {}

    @classmethod
    def configure(cls, log_dir, filename="debug_log.csv", enabled=True, reset=False):
        cls.enabled = enabled
        if not enabled:
            cls.log_path = None
            cls.context = {}
            return None

        os.makedirs(log_dir, exist_ok=True)
        cls.log_path = os.path.join(log_dir, filename)
        cls.context = {}

        if reset and os.path.exists(cls.log_path):
            os.remove(cls.log_path)

        if not os.path.exists(cls.log_path):
            with open(cls.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["timestamp", "tag", "context", "key", "value"],
                )
                writer.writeheader()

        return cls.log_path

    @classmethod
    def set_context(cls, **kwargs):
        if not cls.enabled:
            return
        for key, value in kwargs.items():
            if value is None and key in cls.context:
                del cls.context[key]
            elif value is not None:
                cls.context[key] = value

    @classmethod
    def clear_context(cls):
        cls.context = {}

    @classmethod
    def is_first_batch(cls):
        batch_idx = cls.context.get("batch_idx")
        return batch_idx in (None, 0)

    @classmethod
    def _safe_float(cls, value):
        try:
            return float(value)
        except Exception:
            return None

    @classmethod
    def _summarize_tensor(cls, tensor):
        detached = tensor.detach().cpu()
        summary = {
            "kind": "tensor",
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
        }

        if detached.numel() == 0:
            summary["numel"] = 0
            return summary

        if torch.is_floating_point(detached):
            data = detached.float()
            summary.update(
                {
                    "min": cls._safe_float(data.min()),
                    "max": cls._safe_float(data.max()),
                    "mean": cls._safe_float(data.mean()),
                    "std": cls._safe_float(data.std(unbiased=False)),
                }
            )
        elif detached.dtype == torch.bool:
            summary["true_fraction"] = cls._safe_float(detached.float().mean())
        else:
            data = detached.long()
            summary.update(
                {
                    "min": int(data.min()),
                    "max": int(data.max()),
                }
            )

        if detached.numel() <= 16:
            summary["values"] = detached.reshape(-1).tolist()

        return summary

    @classmethod
    def _summarize_array(cls, array):
        summary = {
            "kind": "ndarray",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
        if array.size == 0:
            summary["numel"] = 0
            return summary

        if np.issubdtype(array.dtype, np.floating):
            arr = array.astype(np.float32, copy=False)
            summary.update(
                {
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                }
            )
        elif np.issubdtype(array.dtype, np.bool_):
            summary["true_fraction"] = float(array.astype(np.float32).mean())
        else:
            summary.update(
                {
                    "min": int(array.min()),
                    "max": int(array.max()),
                }
            )

        if array.size <= 16:
            summary["values"] = array.reshape(-1).tolist()

        return summary

    @classmethod
    def _summarize_value(cls, value):
        if torch.is_tensor(value):
            return cls._summarize_tensor(value)
        if isinstance(value, np.ndarray):
            return cls._summarize_array(value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(k): cls._summarize_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            if len(value) <= 16:
                return [cls._summarize_value(v) for v in value]
            return {
                "kind": type(value).__name__,
                "length": len(value),
                "preview": [cls._summarize_value(v) for v in value[:4]],
            }
        return str(value)

    @classmethod
    def _append_rows(cls, rows):
        if not cls.enabled or cls.log_path is None:
            return

        with open(cls.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["timestamp", "tag", "context", "key", "value"],
            )
            for row in rows:
                writer.writerow(row)

    @classmethod
    def log_debug_csv(cls, tag, *items, **payload):
        if not cls.enabled or cls.log_path is None:
            return

        rows = []
        timestamp = datetime.now().isoformat(timespec="seconds")
        context_json = json.dumps(cls._summarize_value(cls.context), ensure_ascii=True)

        if items:
            payload = dict(payload)
            payload["items"] = list(items)

        if not payload:
            payload = {"event": "triggered"}

        for key, value in payload.items():
            value_json = json.dumps(cls._summarize_value(value), ensure_ascii=True)
            rows.append(
                {
                    "timestamp": timestamp,
                    "tag": str(tag),
                    "context": context_json,
                    "key": str(key),
                    "value": value_json,
                }
            )

        cls._append_rows(rows)

    @classmethod
    def log_debuge_csv(cls, tag, *items, **payload):
        cls.log_debug_csv(tag, *items, **payload)
