"""Offline HTML visualization and session split manifests.
python scripts/visualize_data.py --source documented
python scripts/visualize_data.py --source actual
"""
import argparse
import csv
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_splits import DEV_USERS, TEST_USERS, split_plan

SESSIONS = dict(zip("ABCDEFGHIJKL", [224,156,211,206,126,102,134,138,145,120,148,116]))
WINDOWS = dict(zip(DEV_USERS, [42640,39104,47996,53300,25792,9256,50804,23816]))

def inventory(folder, corr):
    import numpy as np
    sessions, windows, files = {}, {}, {}
    for user in sorted(DEV_USERS + TEST_USERS):
        with np.load(folder / "by_user" / (user + ".npz"), allow_pickle=False) as data:
            names = data["files"]
            gt = data["gt"]
            if gt.shape != (len(names), 1500) or len(names) == 0:
                raise ValueError(f"Invalid gt/files shape for {user}")
            names = [str(n) for n in names]
            if len(set(names)) != len(names) or any(("user" + user + "_") not in n for n in names):
                raise ValueError(f"Duplicate or wrong subject filenames for {user}")
            sessions[user], files[user] = len(names), names
    for user in DEV_USERS:
        path = folder / "windows/dev_cv" / f"{user}_corr{corr}_h200_f25.npz"
        with np.load(path, allow_pickle=False) as data:
            x = data["X"]
            n = len(x)
            if x.shape not in ((n, 200), (n, 200, 1)) or n == 0:
                raise ValueError(f"Invalid X shape: {path}")
            del x
            y = data["y"]
            if y.shape not in ((n, 25), (n, 25, 1)):
                raise ValueError(f"Invalid y shape: {path}")
            windows[user] = n
    return sessions, windows, files

def bars(values, title):
    result = "<h2>" + title + "</h2>"
    for user, n in values.items():
        color = "#a16a16" if user in TEST_USERS else "#087f8c"
        result += f'<div class="bar"><b>{user}</b><span style="width:{n/max(values.values())*70:.2f}%;background:{color}"></span>{n:,}</div>'
    return result

def render(sessions, windows, plan, source):
    origin = ("Số tham chiếu từ docs/CHIA_DU_LIEU.md; chưa đo dữ liệu trên máy."
              if source == "documented" else "Số đo từ NPZ thực tế.")
    output = """<!doctype html><html lang="vi"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>UWB RADAR — Visualize data</title>
<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;background:#f5f8fa;color:#183344}
section{background:white;border-radius:12px;padding:24px;margin:20px 0}
.bar{display:flex;align-items:center;gap:12px;margin:10px 0}.bar b{width:20px}.bar span{height:22px;border-radius:3px}
table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd}
.scroll{overflow:auto}li{margin:12px 0}</style><h1>UWB RADAR · Visualize data</h1>"""
    output += f"<p>{origin}</p><section><h2>Chia theo người</h2><p>{sum(sessions.values()):,} session · Development ABCDEFKL: {sum(sessions[u] for u in DEV_USERS):,} · Test GHIJ: {sum(sessions[u] for u in TEST_USERS):,}</p>"
    output += "<p>Xanh: development. Vàng: test giữ riêng.</p></section><section>"
    output += bars(sessions, "Số session theo người") + bars(windows, "Số cửa sổ train theo người") + "</section>"
    output += '<section><h2>Bốn fold cố định</h2><div class="scroll"><table><tr><th>Fold</th><th>Train</th><th>Validation</th><th>Session train</th><th>Session val</th><th>Cửa sổ train</th></tr>'
    for fold in plan["folds"]:
        cells = [fold["name"], " ".join(fold["train"]), " ".join(fold["validation"]),
                 sum(sessions[u] for u in fold["train"]), sum(sessions[u] for u in fold["validation"]),
                 sum(windows[u] for u in fold["train"])]
        output += "<tr>" + "".join("<td>" + html.escape(str(c)) + "</td>" for c in cells) + "</tr>"
    output += """</table></div><p>Cửa sổ lọc bằng corr với ground truth chỉ dùng cho train.
Validation đọc session radar; ground truth chỉ dùng chấm sau khi đã chọn kênh.</p></section>
<section><h2>Validation → final test → inference</h2><ol>
<li>Train 6 người/fold → chạy radar của 2 người validation → Pearson từng session → trung bình từng người → trung bình 2 người/fold.</li>
<li>Trung bình 4 fold để chọn cấu hình; chạy cùng seed 0, 1, 2. Mỗi người validation đúng một lần mỗi seed.</li>
<li>Chốt cấu hình → train lại đủ ABCDEFKL → checkpoint cuối → test GHIJ.</li>
<li>Inference không phải tập chia thứ tư: radar mới → 120 bin × abs/phase → model chọn kênh → sóng thở. Không cần ground truth.</li>
</ol><p>200 mẫu lịch sử → dự báo 25 mẫu. Pipeline hiện xử lý trọn session 1500 mẫu, chưa phải streaming.</p></section></html>"""
    return output

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["actual", "documented"], default="actual")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/data_visualization")
    parser.add_argument("--corr", type=float, default=0.9)
    args = parser.parse_args()
    if args.source == "documented" and args.corr != 0.9:
        parser.error("Documented counts only support corr=0.9")
    try:
        sessions, windows, files = (inventory(args.data_dir, args.corr) if args.source == "actual"
                                   else (SESSIONS, WINDOWS, {}))
    except (ImportError, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Cannot read actual data: {exc}\nRestore DATA_PREPARE outputs and install numpy, or use --source documented.\n")
    plan = split_plan()
    plan.update(source=args.source, corr=args.corr, sessions=sessions, train_windows=windows)
    # Separate preview and actual output to avoid confusing old manifests with a preview.
    out = args.out / args.source
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(render(sessions, windows, plan, args.source), encoding="utf-8")
    (out / "split_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.source == "actual":
        with (out / "session_splits.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["stage", "role", "user", "npz_path", "session_index", "session_file"])
            stages = [(f["name"], {"train": f["train"], "validation": f["validation"]}) for f in plan["folds"]]
            stages.append(("final", {"train": DEV_USERS, "test": TEST_USERS}))
            for stage, roles in stages:
                for role, users in roles.items():
                    for user in users:
                        for i, name in enumerate(files[user]):
                            writer.writerow([stage, role, user, str((args.data_dir / "by_user" / (user + ".npz")).resolve()), i, name])
    print(out / "index.html")

if __name__ == "__main__":
    main()
