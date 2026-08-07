import ast
from pathlib import Path


def test_parity_and_tuning_scripts_do_not_name_forbidden_scenes():
    for name in ("scripts/run_s5d2_environment_parity.py",):
        text = Path(name).read_text(encoding="utf-8")
        assert "gust_micro_adjust" not in text
        assert "seed_000" not in text


def test_formal_code_does_not_import_frozen_lqr_or_ls_pmpc():
    for name in (
        "src/uav_sway/paper_baseline/sep_nmpc_acados.py",
        "src/uav_sway/paper_baseline/sep_nmpc_controller.py",
    ):
        tree = ast.parse(Path(name).read_text(encoding="utf-8"))
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert all("full_state_lqr" not in item and "da_pmpc" not in item for item in imports)
