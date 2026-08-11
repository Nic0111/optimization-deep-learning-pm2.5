"""Static checks that generated notebooks remain local-first and reproducible."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    ROOT / "assignment1/01_data_acquisition_and_audit.ipynb",
    ROOT / "assignment1/02_eda_and_preprocessing.ipynb",
    ROOT / "assignment2/01_baselines_and_basic_models.ipynb",
    ROOT / "assignment2/02_hyperparameter_tuning.ipynb",
    ROOT / "assignment2/03_final_evaluation.ipynb",
)


def test_notebooks_use_local_kernel_without_colab_references():
    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            if isinstance(cell.get("source", ""), list)
            else cell.get("source", "")
            for cell in notebook["cells"]
        )
        assert notebook["metadata"]["kernelspec"]["name"] == "odl-assignment"
        assert "google.colab" not in source
        assert "/content/drive" not in source
        assert "pip install" not in source
        assert "run_stage(" in source
