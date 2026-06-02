"""Static checks for the inline-only lunar simulation notebook."""

import json
from pathlib import Path

NOTEBOOK = (
    Path(__file__).parents[1] / "notebooks" / "Lunar_Simulation_v000.ipynb"
)


def test_lunar_notebook_is_valid_inline_only_json():
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "savefig" not in source
    assert "LunarCampaign" in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("outputs", []) == []
            assert cell.get("execution_count") is None
