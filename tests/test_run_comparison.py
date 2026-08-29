import csv
import os
import tempfile

from scripts.run_comparison import run


def test_run_produces_all_artifacts():
    with tempfile.TemporaryDirectory() as tmpdir:
        run(output_dir=tmpdir, cfr_iterations=500, cfr_log_every=100, head_to_head_hands=200)

        exploit_path = os.path.join(tmpdir, "exploitability.csv")
        h2h_path = os.path.join(tmpdir, "head_to_head.csv")
        conv_path = os.path.join(tmpdir, "cfr_convergence.csv")
        plot_path = os.path.join(tmpdir, "cfr_convergence.png")

        for path in (exploit_path, h2h_path, conv_path, plot_path):
            assert os.path.exists(path)

        with open(exploit_path) as f:
            rows = list(csv.DictReader(f))
        names = {row["bot"] for row in rows}
        assert names == {"Nash", "Honest", "AlwaysBluff", "Random", "CFR"}

        with open(conv_path) as f:
            conv_rows = list(csv.DictReader(f))
        assert len(conv_rows) == 5  # 500 iterations / log_every=100
