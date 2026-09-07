"""Phân tích vì sao CV quanh 0,74–0,75. KHÔNG train, KHÔNG chạy model.

Chạy từ thư mục gốc đồ án:
    python scripts/analyze_oracle_dev.py
    python scripts/analyze_oracle_dev.py --scores runs /duong/dan/tn1*.zip

Lần sau đã có oracle, chỉ cần đọc thêm ZIP kết quả:
    python scripts/analyze_oracle_dev.py \
        --oracle-csv runs/oracle_dev/oracle_sessions.csv --scores /duong/dan/tn1*.zip

BA CÂU HỎI
1. Oracle all: nhìn GT để chọn tốt nhất trong 120 bin x (abs, phase).
2. Oracle kept: nhìn GT để chọn tốt nhất SAU invert_detector < 0.8.
   all - kept đo phần điểm mất do lọc; kept - model đo phần mất do chọn.
3. Hai model có chọn trùng bin, trùng (bin, method), hay chỉ trùng điểm?

Pearson có dấu, không lấy trị tuyệt đối, không lật dấu sóng để tăng điểm.
Điểm CV = trung bình điểm của từng người A B C D E F K L, rồi trung bình
8 người. Bằng trung bình 4 fold hiện tại vì mỗi fold có đúng 2 người.
Không dùng GHIJ. 0.943 của bài báo KHÔNG phải oracle của dev này.

Đọc CSV trực tiếp hoặc CSV bên trong ZIP, không giải nén checkpoint.
Chỉ nhận scores_<config>_seed<N>_val_<users>.csv. Gộp bản sao giống nhau;
cùng experiment/config/seed/session mà khác kết quả thì DỪNG.
So hai cấu hình theo CÙNG seed, đủ đúng các session dev; thiếu thì báo rõ.

Đầu ra riêng: runs/oracle_dev/{report.md, oracle_sessions.csv,
oracle_users.csv, model_vs_oracle.csv, selection_agreement.csv}.
Không ghi vào summary.csv hay thay đổi các kết quả TN0/TN1/TN2.
"""

import argparse
import csv
import glob
import hashlib
import io
import itertools
import re
import subprocess
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEV_USERS = "ABCDEFKL"
FOLDS = {"AB", "CE", "DF", "KL"}
TOL = 1e-9                  # dung sai khi đọc nhiều bản sao của cùng kết quả
SCORE_TIE = 1e-6            # chỉ là bằng ĐIỂM, không suy ra bằng ứng viên
SCORE_NAME = re.compile(r"scores_(.+)_seed(\d+)_val_([A-Z]+)\.csv$")


# 1. Các phép tính: có thể kiểm riêng bằng dữ liệu nhỏ.
def macro(rows, column):
    """Mỗi người có trọng số bằng nhau, bất kể họ có bao nhiêu buổi."""
    by_user = defaultdict(list)
    for row in rows:
        by_user[row["user"]].append(float(row[column]))
    if not by_user:
        raise ValueError("Không có dữ liệu để tính macro.")
    return float(np.mean([np.mean(values) for values in by_user.values()]))


def pearson_rows(candidates, gt):
    """Pearson từng ứng viên với GT, vector hóa bằng float64 như np.corrcoef."""
    x = np.asarray(candidates, dtype=np.float64)
    y = np.asarray(gt, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean()
    denom = np.sqrt(np.einsum("ij,ij->i", x, x) * np.dot(y, y))
    if not np.isfinite(denom).all() or np.any(denom == 0):
        raise ValueError("Sóng phẳng hoặc dữ liệu không hữu hạn: Pearson không xác định.")
    return np.clip(np.einsum("ij,j->i", x, y) / denom, -1, 1)


def best_indices(scores, kept):
    """Trả chỉ số tốt nhất trước/sau lọc. Điểm hòa thì lấy ứng viên đầu tiên."""
    if not np.isfinite(scores).all():
        raise ValueError("Có Pearson không hữu hạn.")
    surviving = np.flatnonzero(kept)
    if not len(surviving):
        raise ValueError("Bộ lọc loại hết ứng viên: không có oracle sau lọc.")
    return int(np.argmax(scores)), int(surviving[np.argmax(scores[surviving])])


def agreement(a, b):
    """So cùng session, tuyệt đối không ghép theo thứ tự dòng trong CSV."""
    left = {(r["user"], r["session_file"]): r for r in a}
    right = {(r["user"], r["session_file"]): r for r in b}
    if not left or left.keys() != right.keys():
        raise ValueError("Hai cấu hình không có cùng tập session.")
    compared = []
    for key, x in left.items():
        y = right[key]
        same_bin = x["bin"] == y["bin"]
        compared.append({"user": key[0], "same_bin": same_bin,
                         "same_candidate": same_bin and x["method"] == y["method"],
                         "score_tie": abs(x["pearson"] - y["pearson"]) < SCORE_TIE})
    out = {"n_sessions": len(compared)}
    for field in ("same_bin", "same_candidate", "score_tie"):
        out[field + "_count"] = sum(r[field] for r in compared)
        out[field + "_micro"] = float(np.mean([r[field] for r in compared]))
        out[field + "_macro"] = macro(compared, field)
    return out


# 2. Tính oracle từ dữ liệu thật, dùng ĐÚNG biến đổi và bộ lọc của pipeline.
def compute_oracle(data_dir):
    sys.path.insert(0, str(ROOT))
    from src import mobivital_reference as mv

    rows = []
    for user in DEV_USERS:
        path = data_dir / (user + ".npz")
        with np.load(path, allow_pickle=False) as data:
            radar, gt, files = data["uwb"], data["gt"], data["files"]
        if radar.shape != (len(files), 1500, 120) or gt.shape != (len(files), 1500):
            raise ValueError("Sai shape dữ liệu: " + str(path))
        for i, session in enumerate(files):
            try:
                candidates = []
                for bin_number in range(120):
                    transformed = mv.sequence_transforms(radar[i, :, bin_number])
                    candidates.extend([transformed[0], transformed[-1]])
                candidates = np.asarray(candidates)
                kept = np.array([mv.invert_detector(x) < 0.8 for x in candidates])
                scores = pearson_rows(candidates, mv.self_normalize(gt[i]))
                best_all, best_kept = best_indices(scores, kept)
            except ValueError as exc:
                raise ValueError("%s / %s: %s" % (user, session, exc)) from exc
            rows.append({"user": user, "session_file": str(session),
                         "n_candidates": len(scores), "n_kept": int(kept.sum()),
                         "oracle_all": float(scores[best_all]),
                         "oracle_kept": float(scores[best_kept]),
                         "filter_gap": float(scores[best_all] - scores[best_kept]),
                         "all_bin": best_all // 2,
                         "all_method": ("abs", "phase")[best_all % 2],
                         "kept_bin": best_kept // 2,
                         "kept_method": ("abs", "phase")[best_kept % 2]})
        print("Oracle %s: %d buổi, all %.6f, sau lọc %.6f" % (
            user, len(files), macro(rows[-len(files):], "oracle_all"),
            macro(rows[-len(files):], "oracle_kept")), flush=True)
        del radar, gt
    return rows


def load_oracle(path):
    """Dùng lại CSV do script sinh ra, không cần tính lại khi có thêm ZIP."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for field in ("oracle_all", "oracle_kept", "filter_gap"):
            row[field] = float(row[field])
        for field in ("n_candidates", "n_kept", "all_bin", "kept_bin"):
            row[field] = int(row[field])
    return rows


# 3. Đọc lựa chọn đã lưu, KHÔNG nạp model hay chạy lại chấm điểm.
def read_runs(paths):
    runs = defaultdict(dict)
    sources = []

    def add_csv(name, text, source):
        match = SCORE_NAME.fullmatch(Path(name).name)
        if not match:
            return                    # không lấy TN0, test GHIJ, hay summary.csv
        config, seed, fold = match.groups()
        if fold not in FOLDS:
            raise ValueError("Fold không thuộc giao thức dev: " + source)
        experiment = Path(name).parent.name
        if not experiment:
            raise ValueError("CSV phải nằm trong thư mục tên thực nghiệm: " + source)
        key = (experiment, config, int(seed))
        for raw in csv.DictReader(io.StringIO(text)):
            row = {"user": raw["user"], "session_file": raw["session_file"],
                   "bin": int(raw["bin"]), "method": raw["method"],
                   "pearson": float(raw["pearson"])}
            if (row["user"] not in fold or len(row["user"]) != 1
                    or not 0 <= row["bin"] < 120 or row["method"] not in ("abs", "phase")
                    or not np.isfinite(row["pearson"]) or abs(row["pearson"]) > 1 + TOL):
                raise ValueError("Dòng score không hợp lệ: " + source)
            session_key = (row["user"], row["session_file"])
            old = runs[key].get(session_key)
            if old is not None and (old["bin"] != row["bin"]
                                    or old["method"] != row["method"]
                                    or abs(old["pearson"] - row["pearson"]) > TOL):
                raise ValueError("Kết quả trùng khóa nhưng khác nội dung: %s %s ở %s"
                                 % (key, session_key, source))
            runs[key][session_key] = row
        sources.append(source)

    files = set()
    for path in paths:
        path = Path(path)
        if path.is_dir():
            files.update(path.rglob("scores_*.csv"))
            files.update(path.rglob("*.zip"))
        elif path.is_file():
            files.add(path)
        else:
            raise FileNotFoundError(path)
    for path in sorted(files):
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    if SCORE_NAME.fullmatch(Path(member.filename).name):
                        text = archive.read(member).decode("utf-8-sig")
                        add_csv(member.filename, text, str(path) + "::" + member.filename)
        else:
            add_csv(str(path), path.read_text(encoding="utf-8-sig"), str(path))
    return dict(runs), sources


def compare_runs(runs, oracle):
    expected = {(r["user"], r["session_file"]): r for r in oracle}
    complete, model_rows, pair_rows, notes = {}, [], [], []
    for key, sessions in sorted(runs.items()):
        missing = expected.keys() - sessions.keys()
        extra = sessions.keys() - expected.keys()
        if missing or extra:
            notes.append("Bỏ %s / %s / seed %d: thiếu %d, dư %d session so với oracle dev."
                         % (*key, len(missing), len(extra)))
            continue
        rows = list(sessions.values())
        for session, r in sessions.items():
            if r["pearson"] > expected[session]["oracle_kept"] + TOL:
                raise ValueError("Model vượt oracle sau lọc ở %s, %s. Kiểm lại dữ liệu, "
                                 "bộ lọc và nguồn kết quả trước khi so." % (key, session))
        score = macro(rows, "pearson")
        model_rows.append({"experiment": key[0], "config": key[1], "seed": key[2],
                           "n_sessions": len(rows), "cv_macro": score,
                           "oracle_all": macro(oracle, "oracle_all"),
                           "oracle_kept": macro(oracle, "oracle_kept"),
                           "selection_gap": macro(oracle, "oracle_kept") - score})
        complete[key] = rows
    for a, b in itertools.combinations(sorted(complete), 2):
        if a[2] != b[2]:
            continue                  # không so seed 0 của A với seed 1 của B
        pair_rows.append({"model_a": a[0] + "/" + a[1],
                          "model_b": b[0] + "/" + b[1], "seed": a[2],
                          **agreement(complete[a], complete[b])})
    if not complete:
        notes.append("Chưa có bộ scores CV đầy đủ: chưa kết luận được các model chọn trùng nhau.")
    if complete and not pair_rows:
        notes.append("Chưa có hai cấu hình đầy đủ cùng seed để so lựa chọn.")
    return model_rows, pair_rows, notes


# 4. Báo cáo riêng, tên cột nói rõ all/kept và macro/micro.
def write_csv(path, rows, empty_columns):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else empty_columns)
        writer.writeheader()
        writer.writerows(rows)


def report(oracle, users, model_rows, pairs, notes, sources, data_source):
    all_score, kept_score = macro(oracle, "oracle_all"), macro(oracle, "oracle_kept")
    lost = sum(r["filter_gap"] > TOL for r in oracle)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    upstream = subprocess.run(["git", "-C", str(ROOT / "external/mobivital"),
                               "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    lines = ["# Phân tích oracle và lựa chọn kênh trên dev", "",
             "Sinh lúc: %s; commit đồ án: `%s`." % (datetime.now(timezone.utc).isoformat(), commit),
             "SHA256 script (kể cả khi chưa commit): `%s`." % script_hash,
             "Commit MobiVital đang cài: `%s`." % upstream,
             "Nguồn oracle: `%s`." % data_source, "",
             "## 1. Cách hiểu và phạm vi", "",
             "Chỉ ABCDEFKL, %d buổi; không dùng GHIJ. Không train, không chạy model." % len(oracle),
             "Oracle nhìn GT để chọn sóng có Pearson cao nhất; không dự báo hay tái tạo.",
             "All: cả 240 ứng viên. Kept: chỉ ứng viên có invert_detector < 0.8.",
             "Pearson có dấu, không lấy abs(Pearson), không lật sóng.",
             "Điểm chính = trung bình mỗi người, rồi trung bình 8 người (macro), đúng cv_score.",
             "Oracle không có seed. Model báo từng seed; không tự xếp hạng các nhóm thiếu seed.",
             "Số 0.943 của bài báo thuộc phép đánh giá khác, không dùng so trực tiếp ở đây.", "",
             "## 2. Oracle", "",
             "| Người | Buổi | Oracle all | Oracle sau lọc | Mất do lọc |",
             "|---|---:|---:|---:|---:|"]
    for r in users:
        lines.append("| {user} | {n_sessions} | {oracle_all:.6f} | {oracle_kept:.6f} | {filter_gap:.6f} |".format(**r))
    lines += ["| **Macro 8 người** | %d | **%.6f** | **%.6f** | **%.6f** |"
              % (len(oracle), all_score, kept_score, all_score - kept_score), "",
              "Bộ lọc làm giảm điểm oracle ở %d/%d buổi (chênh > 1e-9)." % (lost, len(oracle)),
              "Số ứng viên còn lại: min %d, trung bình %.2f, max %d."
              % (min(r["n_kept"] for r in oracle), np.mean([r["n_kept"] for r in oracle]),
                 max(r["n_kept"] for r in oracle)), "",
              "Trung vị oracle sau lọc trên các buổi (không phải macro): %.6f."
              % np.median([r["oracle_kept"] for r in oracle])]
    for threshold in (0.3, 0.5):
        count = sum(r["oracle_kept"] < threshold for r in oracle)
        lines.append("Buổi có oracle sau lọc < %.1f: %d/%d (%.2f%%, micro)."
                     % (threshold, count, len(oracle), 100 * count / len(oracle)))
    lines += ["", "| Fold | Oracle all (macro 2 người) | Oracle kept (macro 2 người) |",
              "|---|---:|---:|"]
    for fold in ("AB", "CE", "DF", "KL"):
        rows = [r for r in oracle if r["user"] in fold]
        lines.append("| val_%s | %.6f | %.6f |"
                     % (fold, macro(rows, "oracle_all"), macro(rows, "oracle_kept")))
    lines += ["", "## 3. Điểm model so với oracle", "",
              "Chỉ tính khi đủ đúng mọi session dev; cv_macro được tính lại từ scores CSV.", "",
              "| Cấu hình | Seed | CV macro | Oracle kept − model |",
              "|---|---:|---:|---:|"]
    for r in model_rows:
        lines.append("| {experiment}/{config} | {seed} | {cv_macro:.6f} | {selection_gap:.6f} |".format(**r))
    lines += ["", "## 4. Chọn trùng hay chỉ bằng điểm?", "",
              "Ghép cùng seed và cùng (user, session_file). Cùng bin khác method vẫn là khác sóng.",
              "Tỷ lệ macro: trung bình tỷ lệ từng người. Micro: tỷ lệ trên toàn bộ buổi.",
              "CSV ghi cả số buổi và hai tỷ lệ; bảng dưới dùng macro.", "",
              "| A | B | Seed | Trùng bin | Trùng (bin, method) | Bằng điểm (<1e-6) |",
              "|---|---|---:|---:|---:|---:|"]
    for r in pairs:
        lines.append("| %s | %s | %d | %.2f%% | %.2f%% | %.2f%% |" % (
            r["model_a"], r["model_b"], r["seed"], 100*r["same_bin_macro"],
            100*r["same_candidate_macro"], 100*r["score_tie_macro"]))
    lines += ["", "## 5. Giới hạn và dữ liệu còn thiếu", ""]
    lines.extend("- " + note for note in notes)
    lines += ["- Điểm oracle là trung bình, không có nghĩa mọi buổi đều có sóng tốt.",
              "- Khoảng cách tới oracle không bảo đảm đổi kiến trúc sẽ thu hồi được.",
              "- Trùng lựa chọn chỉ mô tả hành vi; chưa chứng minh nguyên nhân hay ý nghĩa thống kê.",
              "- Chưa đánh giá sai số số lần thở/phút trong script này.",
              "- Nếu dùng --oracle-csv, phải là kết quả cùng dữ liệu và cùng phiên bản bộ lọc.",
              "", "## 6. Nguồn CSV/ZIP đã đọc", ""]
    lines.extend("- `%s`" % s for s in sources)
    if not sources:
        lines.append("Chưa tìm thấy scores CV phù hợp. Đưa ZIP vào --scores để bổ sung.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/by_user")
    parser.add_argument("--scores", nargs="*", default=[str(ROOT / "runs")],
                        help="CSV, ZIP hoặc thư mục chứa chúng; hỗ trợ glob có dấu nháy.")
    parser.add_argument("--oracle-csv", type=Path, help="Dùng lại oracle_sessions.csv đã tính.")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/oracle_dev")
    args = parser.parse_args()
    paths = []
    for pattern in args.scores:
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError("Không tìm thấy nguồn scores: " + pattern)
        paths.extend(Path(p) for p in matches)
    runs, sources = read_runs(paths)
    oracle = load_oracle(args.oracle_csv) if args.oracle_csv else compute_oracle(args.data_dir)
    keys = {(r["user"], r["session_file"]) for r in oracle}
    if len(keys) != len(oracle) or {r["user"] for r in oracle} != set(DEV_USERS):
        raise ValueError("Oracle trùng session hoặc không có đúng 8 người dev.")
    for r in oracle:
        if (not all(np.isfinite(r[c]) for c in ("oracle_all", "oracle_kept", "filter_gap"))
                or not -1 <= r["oracle_kept"] <= r["oracle_all"] <= 1
                or r["n_candidates"] != 240 or not 1 <= r["n_kept"] <= 240
                or abs(r["filter_gap"] - (r["oracle_all"] - r["oracle_kept"])) > TOL):
            raise ValueError("Dòng oracle không hợp lệ: " + str(r))
    users = []
    for user in DEV_USERS:
        rows = [r for r in oracle if r["user"] == user]
        users.append({"user": user, "n_sessions": len(rows),
                      **{c: macro(rows, c) for c in ("oracle_all", "oracle_kept", "filter_gap")}})
    model_rows, pairs, notes = compare_runs(runs, oracle)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows, columns in [
        ("oracle_sessions", oracle, []), ("oracle_users", users, []),
        ("model_vs_oracle", model_rows, ["experiment", "config", "seed", "cv_macro", "selection_gap"]),
        ("selection_agreement", pairs, ["model_a", "model_b", "seed", "same_bin_macro", "same_candidate_macro"])]:
        write_csv(args.output / (name + ".csv"), rows, columns)
    text = report(oracle, users, model_rows, pairs, notes, sources, args.oracle_csv or args.data_dir)
    (args.output / "report.md").write_text(text, encoding="utf-8")
    print("\nOracle dev macro: all %.6f; sau lọc %.6f; mất do lọc %.6f" % (
        macro(oracle, "oracle_all"), macro(oracle, "oracle_kept"), macro(oracle, "filter_gap")))
    for note in notes:
        print(note)
    print("Báo cáo:", args.output / "report.md")


if __name__ == "__main__":
    main()
