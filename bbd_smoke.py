"""One Red-Green row per scheme (rep 99) as a smoke test; run with `python bbd_smoke.py`."""
import json
import os
import bbd
bbd.SMOKE = os.environ.get("SMOKE", "").split(",") if os.environ.get("SMOKE") else None

if __name__ == "__main__":
    with bbd.app.run():
        ids = bbd.list_prompt_ids.remote([bbd.RG_LIST], [4])
        row = {str(d): ids[f"{bbd.RG_LIST}|I ate|{d}|4"] for d in range(1, 10)}
        calls = {s: (bbd.rg_row_prc.spawn(4, "I ate", 99, row) if s == "prc"
                     else bbd.rg_row_hf.spawn(s, 4, "I ate", 99, row)) for s in (bbd.SMOKE or bbd.SCHEMES)}
        for scheme, call in calls.items():
            try:
                path = call.get()
                data = json.loads(b"".join(bbd.results.read_file(path.removeprefix("/results/"))))
                cells = data["cells"]
                print(scheme, "valid", [c["valid"] for c in cells.values()], "drawn",
                      sum(c["drawn"] for c in cells.values()), "counts d1", cells["1"]["counts"],
                      "|", cells["1"]["examples"][:1])
            except Exception as error:
                print(scheme, "FAILED", repr(error)[-600:])
