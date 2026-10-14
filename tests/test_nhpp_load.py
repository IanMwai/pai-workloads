import csv
import importlib.util
import random
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "nhpp_load", Path(__file__).resolve().parents[1] / "pai_workloads" / "nhpp_load.py")
nhpp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(nhpp)


def test_thinning_matches_integrated_rate():
    lam = nhpp.sinusoid_profile(0.05, 0.5, 14.0)
    arrivals = nhpp.thinning(lam, 86400.0, random.Random(1))
    expected = 0.275 * 86400  # mean of the cosine profile
    assert abs(len(arrivals) - expected) < 0.05 * expected
    assert arrivals == sorted(arrivals) and arrivals[-1] < 86400.0


def test_compression_keeps_count_and_shape():
    args = nhpp.build_parser().parse_args(["--hours", "24", "--compress", "12", "--seed", "3"])
    _, _, arrivals = nhpp.plan(args)
    assert arrivals[-1] < 2 * 3600  # a day in two wall hours
    per_hour = [0] * 24
    for t in arrivals:
        per_hour[min(int(t * 12 // 3600), 23)] += 1
    assert per_hour[14] > 3 * per_hour[2]  # afternoon peak survives compression


def test_trace_day_rates_and_tokens(tmp_path):
    path = tmp_path / "trace.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["TIMESTAMP", "ContextTokens", "GeneratedTokens"])
        for minute in range(0, 1440, 2):  # one request every 2 min on day 0
            w.writerow([f"2023-11-16 {minute // 60:02d}:{minute % 60:02d}:00.1234567", 100, 20])
        w.writerow(["2023-11-17 01:00:00.0000000", 5, 5])  # day 1, excluded
    lam, tokens, date, n = nhpp.load_trace(path, 0, 1.0)
    assert date == "2023-11-16" and n == 720
    assert lam(0.0) == 1 / 60 and lam(60.0) == 0.0
    assert tokens and all(t == (100, 20) for t in tokens)


def test_dry_run_prints_hours(capsys):
    assert nhpp.main(["--dry-run", "--hours", "3", "--compress", "3"]) == 0
    out = capsys.readouterr().out
    assert "hour  0:" in out and "hour  2:" in out
