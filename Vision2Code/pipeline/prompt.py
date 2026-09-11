"""Build (system, user_text, image_paths) for image → CadQuery code.

Native records render a composite lazily from ``step_path``. Custom training
records may instead provide ``images`` or ``composite_png`` directly.
"""

from __future__ import annotations

from pathlib import Path

SYSTEM_PROMPT = """You are an expert CAD engineer. Given a 2x2 composite of 4 diagonal views of a mechanical part, write a CadQuery Python program that reproduces the geometry.

Views (cameras at, all looking at part center [0.5, 0.5, 0.5]):
- Top-left:     [-1, -1, -1]
- Top-right:    [ 1,  1,  1]
- Bottom-left:  [ 1, -1,  1]
- Bottom-right: [-1,  1, -1]

Renders are normalized: bbox centered at [0.5, 0.5, 0.5], longest side maps to [0,1]. Match orientation exactly — world XYZ in your code must match world XYZ in the renders.

Output ONLY a single ```python fenced block:
- start with `import cadquery as cq`
- store the final solid in `result`
- no prose, no comments outside the fence"""

USER_PROMPT = (
    "Generate CadQuery code to recreate this industrial part shown in the "
    "4-view composite render."
)


def build(record: dict, data_dir: Path) -> tuple[str, str, list[Path]]:
    def resolve(value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else data_dir / path

    image_values = record.get("images") or []
    if isinstance(image_values, (str, Path)):
        image_values = [image_values]
    if image_values:
        image_paths = [resolve(value) for value in image_values]
    elif record.get("composite_png"):
        image_paths = [resolve(record["composite_png"])]
    elif any(record.get(f"view_{index}_png") for index in range(4)):
        image_paths = [
            resolve(record[f"view_{index}_png"])
            for index in range(4)
            if record.get(f"view_{index}_png")
        ]
    else:
        from benchcad_core.scoring.views import composite_for_step

        step = resolve(record["step_path"])
        image_paths = [composite_for_step(step)]

    user_text = record.get("user_text") or USER_PROMPT
    return SYSTEM_PROMPT, str(user_text), image_paths
