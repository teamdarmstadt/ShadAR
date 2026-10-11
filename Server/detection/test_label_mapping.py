"""
Check for the COCO label mapping — the piece that lets the headset's detector toggle work.

The shader filters on SLOT indices from its own `@detect-classes` list. YOLOE is told that
list via set_classes(), so its class index already is the slot. yolo26-seg is closed COCO and
knows nothing about the shader, so its COCO id has to be mapped back onto the slots.

The hard part is vocabulary, not indices: a shader asks for "television screen", COCO answers
"tv". Those share no substring, which is exactly how COCO mode first shipped returning zero
detections. Matching is therefore word-based plus a synonym table.

Run:  python Server/detection/test_label_mapping.py
"""

import ast
import sys
from pathlib import Path

SERVER = Path(__file__).with_name("server.py")

# Pulled out of server.py rather than imported: importing would drag in torch/ultralytics and
# load models, which this check has no business doing.
WANTED = {"_words", "_phrase_matches", "build_coco_slot_map", "build_yoloe_prompts"}


def load_mapping():
    tree = ast.parse(SERVER.read_text(encoding="utf-8"))
    body = [
        n for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in WANTED)
        or (isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in ("COCO_SYNONYMS", "COCO_NAME_TO_ID")
            for t in n.targets))
    ]
    found = {n.name for n in body if isinstance(n, ast.FunctionDef)}
    assert WANTED <= found, f"missing from server.py: {WANTED - found}"

    namespace = {"re": __import__("re"), "logger": _QuietLogger()}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SERVER), "exec"), namespace)
    return namespace


class _QuietLogger:
    """build_coco_slot_map logs its mapping; the check only cares about the return value."""
    def info(self, *_a, **_k): pass
    def warning(self, *_a, **_k): pass
    def error(self, *_a, **_k): pass


NS = load_mapping()
COCO = {v: k for k, v in NS["COCO_NAME_TO_ID"].items()}


def slots(vocabulary, coco_classes=None):
    """{coco class id -> shader slot} for a requested vocabulary."""
    return NS["build_coco_slot_map"](vocabulary, COCO, coco_classes)


def main():
    name_to_id = NS["COCO_NAME_TO_ID"]
    tv, laptop, cup, chair, person = (
        name_to_id["tv"], name_to_id["laptop"], name_to_id["cup"],
        name_to_id["chair"], name_to_id["person"],
    )

    # The reported failure: a shader asking for monitors found nothing, because "tv" is not a
    # substring of "television screen". Both phrases must now reach COCO's "tv".
    m = slots(["television screen", "computer monitor"])
    assert m.get(tv) == 0, f"'television screen' must map to tv, got {m}"

    # A COCO class can serve only one slot (a detection carries one label_id), so when two
    # phrases resolve to the same class the EARLIEST slot wins — deterministic, not arbitrary.
    # The second slot stays empty; the server logs that as a collision, not as a missing
    # synonym, because the two need opposite fixes.
    m = slots(["computer monitor", "television screen"])
    assert m.get(tv) == 0
    assert list(m.values()).count(1) == 0

    # Plain names and natural phrases both work, in the shader's slot order.
    m = slots(["chair", "laptop"])
    assert m.get(chair) == 0 and m.get(laptop) == 1

    # A COCO label named inside a longer phrase still matches ("cup" in "red coffee cup").
    assert slots(["red coffee cup"]).get(cup) == 0

    # ...and the synonym table covers the reverse direction ("mug" -> cup).
    assert slots(["mug"]).get(cup) == 0

    # Word-based matching must NOT fire on a shared substring: "car" inside "cardboard".
    assert name_to_id["car"] not in slots(["cardboard box"])

    # Anything the shader did not ask for is absent, so the caller drops it.
    assert person not in slots(["chair"])

    # A vocabulary with no COCO equivalent maps nothing — the caller drops everything and the
    # server warns. Silently returning raw COCO ids here would light up the wrong objects.
    assert slots(["dragon", "unicorn"]) == {}

    # Empty / junk vocabularies must not raise.
    assert slots([]) == {}
    assert slots(["", "   "]) == {}

    # --- the shader's own @detect-coco list (written by the LLM at generation time) ---

    # It wins over name matching, which is the whole point: the LLM knew the intent.
    assert slots(["glowing rectangle"], ["tv"]).get(tv) == 0

    # "-" / empty means the LLM found no equivalent: that slot maps to nothing, and the
    # phrase must NOT quietly fall back to name matching, or the LLM's decision is moot.
    assert slots(["dragon", "chair"], ["", "chair"]) == {chair: 1}

    # A hallucinated COCO class falls back to name matching for that slot rather than
    # poisoning the map.
    assert slots(["chair"], ["comfy_seat_9000"]).get(chair) == 0

    # A list whose length does not line up is discarded wholesale — a shifted list would
    # point every slot at the wrong class, which is worse than guessing.
    assert slots(["chair", "laptop"], ["laptop"]).get(chair) == 0

    # No list at all: unchanged heuristic behaviour, so old shaders keep working.
    assert slots(["television screen"], None).get(tv) == 0
    assert slots(["television screen"], []).get(tv) == 0

    # --- YOLOE: each slot's COCO name rides along as an extra prompt for the same slot ---
    yp = NS["build_yoloe_prompts"]

    # The reported failure: "computer monitor" finds nothing on YOLOE, "tv" does. The alias is
    # appended AFTER the phrases, so every phrase keeps its slot index as its prompt index.
    assert yp(["computer monitor"], ["tv"]) == (["computer monitor", "tv"], [0, 0])
    assert yp(["guitar", "red mug"], ["", "cup"]) == (["guitar", "red mug", "cup"], [0, 1, 1])

    # No duplicate prompt when the phrase already is the COCO name.
    assert yp(["person"], ["person"]) == (["person"], [0])

    # "-", unknown names, a shared class (earliest slot wins) and a misaligned list add nothing.
    assert yp(["human face"], ["-"]) == (["human face"], [0])
    assert yp(["chair"], ["comfy_seat_9000"]) == (["chair"], [0])
    assert yp(["screen", "monitor"], ["tv", "tv"]) == (["screen", "monitor", "tv"], [0, 1, 0])
    assert yp(["chair", "laptop"], ["laptop"]) == (["chair", "laptop"], [0, 1])
    assert yp(["chair"], None) == (["chair"], [0])

    print("COCO label mapping: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
