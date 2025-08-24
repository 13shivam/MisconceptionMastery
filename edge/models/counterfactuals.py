from typing import Dict


def generate_counterfactual_item(original_item: Dict, misconception: int) -> Dict:
    stem = original_item.get("stem", "Item")
    options = original_item.get("distractors", ["A", "B", "C", "D"])
    if isinstance(options, str):
        options = options.split("|")
    cf = {
        "stem": f"{stem} (Consider edge-case #{misconception} explicitly.)",
        "options": options[::-1],
        "correct_index": 0
    }
    return cf
