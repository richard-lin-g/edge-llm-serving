#!/usr/bin/env python3
"""bench/d8_gguf_meta.py — 直接从 GGUF 头里读架构参数，算 KV Cache 每 token 字节数

为什么不用猜：KV Cache 的大小只取决于四个量（层数、KV 头数、head_dim、精度），
这四个量全写在 GGUF 的 metadata 里。读出来，公式才有资格和实测对表。

用法：
  python3 bench/d8_gguf_meta.py ~/models/qwen2.5-3b-instruct-q4_k_m.gguf \\
      --out results/d8_kv_model.json

输出：JSON（层数/头数/KV 头数/head_dim/每 token 字节数）+ 一张 ctx→MiB 表。
退出码：读不到关键字段（block_count / head_count）返回 1。
"""
import argparse
import json
import os
import struct
import sys

# GGUF metadata value types
U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)

SCALARS = {
    U8: ("<B", 1), I8: ("<b", 1), U16: ("<H", 2), I16: ("<h", 2),
    U32: ("<I", 4), I32: ("<i", 4), F32: ("<f", 4), BOOL: ("<?", 1),
    U64: ("<Q", 8), I64: ("<q", 8), F64: ("<d", 8),
}

WANT = (
    ".block_count", ".attention.head_count", ".attention.head_count_kv",
    ".embedding_length", ".attention.key_length", ".attention.value_length",
    ".context_length",
)


def read_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")


def read_val(f, t):
    if t == STR:
        return read_str(f)
    if t == ARR:
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        if et == STR:                      # 词表：只跳过，不取值
            for _ in range(n):
                ln = struct.unpack("<Q", f.read(8))[0]
                f.seek(ln, 1)
            return None
        if et not in SCALARS:              # 嵌套数组（极少见）——跳过
            f.seek(0) if False else None
            raise ValueError("nested array not supported")
        fmt, size = SCALARS[et]
        if n <= 64:                        # head_count_kv 可能是按层数组
            return [struct.unpack(fmt, f.read(size))[0] for _ in range(n)]
        f.seek(size * n, 1)
        return None
    if t not in SCALARS:
        raise ValueError(f"unknown value type {t}")
    fmt, size = SCALARS[t]
    return struct.unpack(fmt, f.read(size))[0]


def parse(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError("不是 GGUF 文件（magic 不对）")
        version = struct.unpack("<I", f.read(4))[0]
        _tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        meta = {}
        arch = None
        for _ in range(n_kv):
            key = read_str(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            val = read_val(f, vtype)
            if key == "general.architecture":
                arch = val
            meta[key] = val
    return version, arch, meta


def pick(meta, arch, suffix):
    for prefix in (arch, None):
        k = (prefix + suffix) if prefix else None
        if k and k in meta and meta[k] is not None:
            return meta[k]
        # 兜底：按后缀模糊匹配
        for kk, vv in meta.items():
            if kk.endswith(suffix) and vv is not None:
                return vv
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--out", default="results/d8_kv_model.json")
    ap.add_argument("--bytes-per-element", type=float, default=2.0,
                    help="KV 精度：f16=2.0（默认）；q8_0≈1.0625；q4_0≈0.5625")
    a = ap.parse_args()

    try:
        version, arch, meta = parse(a.gguf)
    except Exception as e:
        print(f"! 读 GGUF 失败：{e}", file=sys.stderr)
        return 1

    layers = pick(meta, arch, ".block_count")
    heads = pick(meta, arch, ".attention.head_count")
    kvmode = pick(meta, arch, ".attention.head_count_kv")
    emb = pick(meta, arch, ".embedding_length")
    klen = pick(meta, arch, ".attention.key_length")
    vlen = pick(meta, arch, ".attention.value_length")
    train_ctx = pick(meta, arch, ".context_length")

    if isinstance(heads, list):
        heads = max(heads) if heads else None
    if isinstance(kvmode, list):
        kvmode = max(kvmode) if kvmode else None
    if kvmode is None:                      # 没有 GQA 字段 = 标准 MHA
        kvmode = heads

    if not layers or not heads or not emb:
        print(f"! 关键字段缺失（layers={layers} heads={heads} emb={emb}）", file=sys.stderr)
        return 1

    head_dim = int(klen) if klen else int(emb) // int(heads)
    v_dim = int(vlen) if vlen else head_dim

    # KV Cache 每 token 字节：K 与 V 各一份
    per_token = int(layers) * int(kvmode) * (head_dim + v_dim) * a.bytes_per_element

    out = {
        "model": os.path.basename(a.gguf),
        "gguf_version": version,
        "arch": arch,
        "layers": int(layers),
        "heads": int(heads),
        "kv_heads": int(kvmode),
        "head_dim": head_dim,
        "value_dim": v_dim,
        "bytes_per_element": a.bytes_per_element,
        "kv_bytes_per_token": int(round(per_token)),
        "kv_kib_per_token": round(per_token / 1024, 2),
        "train_context": train_ctx,
        "gqa_ratio": round(int(heads) / int(kvmode), 2) if kvmode else None,
        "ctx_table_mib": {
            str(c): round(per_token * c / (1024 * 1024), 1)
            for c in (512, 1024, 2048, 4096, 8192, 16384, 32768)
        },
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"== {out['model']} ==")
    print(f"arch={arch}  layers={layers}  heads={heads}  kv_heads={kvmode}"
          f"  head_dim={head_dim}  (GQA {out['gqa_ratio']}:1)")
    print(f"KV 每 token = {layers} × {kvmode} × ({head_dim}+{v_dim}) × {a.bytes_per_element} B"
          f" = {per_token:,.0f} B = {out['kv_kib_per_token']} KiB")
    print()
    print("ctx      KV Cache（MiB）")
    for c, m in out["ctx_table_mib"].items():
        print(f"{c:>6}   {m:>8.1f}")
    if train_ctx:
        print(f"\n! 注意：模型训练上下文 {train_ctx}，按此分配需 "
              f"{round(per_token * int(train_ctx) / (1024 * 1024), 0):.0f} MiB KV——4 GB 卡上别用默认 ctx")
    print(f"\n已写 {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
