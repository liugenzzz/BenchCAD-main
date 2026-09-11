from pathlib import Path

from Vision2Code.pipeline.prompt import build


def test_build_uses_custom_images_and_conversation_prompt(tmp_path):
    record = {
        "step_path": "unused.step",
        "images": ["renders/composite.png"],
        "user_text": "Generate the CADQuery code. Just the code.",
    }

    _, user_text, image_paths = build(record, tmp_path)

    assert user_text == "Generate the CADQuery code. Just the code."
    assert image_paths == [tmp_path / "renders" / "composite.png"]


def test_build_preserves_absolute_composite_path(tmp_path):
    image = (tmp_path / "absolute" / "composite.png").resolve()
    record = {"step_path": "unused.step", "composite_png": image}

    _, _, image_paths = build(record, tmp_path)

    assert image_paths == [image]
