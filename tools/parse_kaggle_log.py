# -*- coding: utf-8 -*-
"""Turn a downloaded Kaggle kernel log into the tables the paper needs.

The Kaggle CLI can fetch a run's log even when it refuses to serve the run's
output files, so the log is the reliable record of every number. This parses
it into: the per-config result table, the tone confusion matrices, and the
training curves.

    python tools/parse_kaggle_log.py run1.log run2.log --out docs/ket_qua.md
"""
import argparse
import json
import os
import re

TONES = ["level", "acute", "grave", "hook", "tilde", "dot"]
TONE_VN = {"level": "ngang", "acute": "sắc", "grave": "huyền",
           "hook": "hỏi", "tilde": "ngã", "dot": "nặng"}


def flatten(path):
    """Kaggle logs are JSON arrays of {stream_name, time, data}."""
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    try:
        rec = json.loads(raw)
        return "".join(e.get("data", "") for e in rec), max(
            (e.get("time", 0) for e in rec), default=0)
    except json.JSONDecodeError:
        return raw, 0


def parse_evals(txt):
    """Every named block followed by a 'split=test' report.

    Round 1 evaluated in a dedicated cell and printed '===== name =====';
    round 2 evaluates inside the training loop under '##### name (da chay
    Xh) #####'. Both delimiters have to be recognised or a whole run parses
    as zero configs.
    """
    out = []
    parts = re.split(r"(?:={5,}\s*(\w+)\s*={5,}"
                     r"|#{5,}\s*(\w+)\s+\(da chay [\d.]+h\)\s*#{5,})", txt)
    # re.split with two groups yields [pre, g1, g2, body, g1, g2, body, ...]
    merged = [parts[0]]
    for i in range(1, len(parts), 3):
        merged.append(parts[i] or parts[i + 1])
        merged.append(parts[i + 2])
    parts = merged
    for i in range(1, len(parts), 2):
        name, body = parts[i], parts[i + 1]
        if "split=test" not in body:
            continue
        g = lambda p: (float(m.group(1))
                       if (m := re.search(p, body)) else float("nan"))
        gi = lambda p: (int(m.group(1))
                        if (m := re.search(p, body)) else 0)
        row = {
            "name": name,
            "n": gi(r"split=test\s+n=(\d+)"),
            "word": g(r"word acc\s+:\s+([\d.]+)"),
            "word_ci": g(r"word acc \(ci\)\s+:\s+([\d.]+)"),
            "ned": g(r"1 - NED\s+:\s+([\d.]+)"),
        }
        for key, pat in [("any", r"co dau bat ky\s*:\s*([\d.]+)"),
                         ("multi", r">=2 ky tu co dau\s*:\s*([\d.]+)"),
                         ("conf", r"chua hoi/nga\s*:\s*([\d.]+)"),
                         ("stack", r"co ky tu chong dau\s*:\s*([\d.]+)")]:
            row[key] = g(pat)
        # old-format logs only carry the useless n=62 "heavy" number
        if row["any"] != row["any"]:
            row["heavy_old"] = g(r"diacritic-heavy\s+:\s+([\d.]+)")
        rows = re.findall(
            r"^\s*(level|acute|grave|hook|tilde|dot)\s+((?:\d+\s+){5}\d+)\s*$",
            body, re.M)
        if len(rows) == 6:
            row["conf_matrix"] = {r[0]: [int(x) for x in r[1].split()]
                                  for r in rows}
        row["errors"] = gi(r"\((\d+) errors listed\)")
        out.append(row)
    return out


def parse_curves(txt):
    """(epoch, loss, word acc) per training stage, split where epoch resets."""
    eps = re.findall(
        r"ep\s+(\d+) \| loss ([\d.]+).*?\| word (nan|[\d.]+).*?\| (\d+)s", txt)
    stages, cur, prev = [], [], -1
    for e, loss, word, sec in eps:
        e = int(e)
        if e <= prev and cur:
            stages.append(cur)
            cur = []
        cur.append((e, float(loss), word, int(sec)))
        prev = e
    if cur:
        stages.append(cur)
    return stages


def fmt_table(rows):
    has_new = any("any" in r and r["any"] == r["any"] for r in rows)
    head = "| Cấu hình | word acc | word acc (ci) | 1−NED |"
    sep = "|---|---|---|---|"
    if has_new:
        head += " có dấu | ≥2 dấu | hỏi/ngã | chồng dấu |"
        sep += "---|---|---|---|"
    head += " lỗi |"
    sep += "---|"
    lines = [head, sep]
    for r in rows:
        c = f"| `{r['name']}` | {r['word']:.4f} | {r['word_ci']:.4f} | {r['ned']:.4f} |"
        if has_new:
            for k in ["any", "multi", "conf", "stack"]:
                v = r.get(k, float("nan"))
                c += f" {v:.4f} |" if v == v else " – |"
        c += f" {r['errors']} |"
        lines.append(c)
    return "\n".join(lines)


def fmt_confusion(r):
    M = r.get("conf_matrix")
    if not M:
        return ""
    lines = [f"**{r['name']}** — ma trận nhầm thanh điệu (hàng = đúng, cột = dự đoán)",
             "", "| |" + "|".join(TONE_VN[t] for t in TONES) + "| đúng |",
             "|---|" + "---|" * (len(TONES) + 1)]
    for i, t in enumerate(TONES):
        row = M[t]
        acc = 100 * row[i] / max(sum(row), 1)
        lines.append(f"| **{TONE_VN[t]}** |" + "|".join(str(v) for v in row)
                     + f"| {acc:.1f}% |")
    h2t, t2h = M["hook"][4], M["tilde"][3]
    lines.append("")
    lines.append(f"Nhầm hỏi↔ngã: hỏi→ngã {h2t}, ngã→hỏi {t2h}, **tổng {h2t + t2h}**")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    labels = args.labels or [os.path.basename(p) for p in args.logs]

    doc = ["# Kết quả VietTDR", ""]
    for path, label in zip(args.logs, labels):
        txt, dur = flatten(path)
        rows = parse_evals(txt)
        stages = parse_curves(txt)
        doc += [f"## {label}", "",
                f"Thời lượng phiên: **{dur / 3600:.2f} giờ**. "
                f"Số cấu hình đánh giá: **{len(rows)}**.", ""]
        if rows:
            doc += [fmt_table(rows), ""]
            base = next((r for r in rows if r["name"] == "full"), None)
            if base and len(rows) > 1:
                doc += ["Chênh lệch so với `full` (điểm phần trăm):", ""]
                for r in rows:
                    if r["name"] == "full":
                        continue
                    doc.append(f"- `{r['name']}`: word "
                               f"{100 * (base['word'] - r['word']):+.2f}, "
                               f"1−NED {100 * (base['ned'] - r['ned']):+.2f}")
                doc.append("")
            for r in rows:
                c = fmt_confusion(r)
                if c:
                    doc += [c, ""]
        if stages:
            doc += ["Thời gian huấn luyện từng chặng:", ""]
            for i, st in enumerate(stages):
                tt = sum(x[3] for x in st)
                last = [x for x in st if x[2] != "nan"]
                acc = f", val cuối {last[-1][2]}" if last else ""
                doc.append(f"- chặng {i + 1}: {len(st)} epoch, "
                           f"{tt / len(st):.0f}s/epoch, tổng {tt / 3600:.2f}h{acc}")
            doc.append("")

    text = "\n".join(doc)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
