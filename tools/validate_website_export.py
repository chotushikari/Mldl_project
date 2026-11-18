"""Reject missing artifacts or invented combinations in the partial site export."""
import json
from pathlib import Path


root = Path("outputs")
payload = json.loads((root / "website_results_partial.json").read_text(encoding="utf-8"))
assert payload["status"] == "partial_experiments"
assert len(payload["experiments"]) >= 17
ids = [experiment["id"] for experiment in payload["experiments"]]
assert len(set(ids)) == len(ids)
for experiment in payload["experiments"]:
    assert experiment["dataset"] in ("ACRIMA", "RIM-ONE r3", "ORIGA", "REFUGE", "DRISHTI-GS1")
    assert not (experiment["dataset"] == "ACRIMA" and experiment["task"] == "segmentation")
    assert len(experiment["predictions"]) == experiment["metrics"]["n_test"]
    for field in ("gradcam_panels", "example_panels"):
        for path in experiment.get(field, []):
            assert Path(path).is_file(), path
for path in payload["assets"].values():
    assert Path(path).is_file(), path
print("Validated", len(ids), "measured experiment combinations:", ", ".join(ids))
