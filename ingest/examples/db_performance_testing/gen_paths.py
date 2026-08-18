#!/usr/bin/env python3
"""Generate N unique random image paths to a text file.

UUID4 hex strings are used as filenames — collision probability at 12M is ~8.6e-9,
effectively unique without a seen-set.
"""
import argparse
import time
import uuid


def main():
    p = argparse.ArgumentParser(description="Generate random image paths")
    p.add_argument("--n", type=int, default=12_000_000, help="Number of paths (default: 12M)")
    p.add_argument("--out", default="image_paths.txt", help="Output file (default: image_paths.txt)")
    args = p.parse_args()

    print(f"Generating {args.n:,} paths → {args.out}")
    t0 = time.time()
    with open(args.out, "w", buffering=1 << 20) as f:  # 1 MB write buffer
        for _ in range(args.n):
            uid = uuid.uuid4().hex
            f.write(f"/data/images/{uid[:4]}/{uid[4:8]}/{uid}.jpg\n")
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s  ({args.n / elapsed:,.0f} paths/s)")


if __name__ == "__main__":
    main()
