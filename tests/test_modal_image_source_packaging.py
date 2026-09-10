from pathlib import Path


SOURCE = Path("scripts/modal_open_models.py")


def _function_decorator(source: str, function_name: str) -> str:
    marker = f"def {function_name}("
    function_index = source.index(marker)
    decorator_index = source.rfind("@app.function", 0, function_index)
    assert decorator_index >= 0
    return source[decorator_index:function_index]


def test_all_custom_modal_images_package_clipper_source() -> None:
    source = SOURCE.read_text(encoding="utf-8")

    media_block = source.split("media_image = (", 1)[1].split("\nstate_image =", 1)[0]
    assert '.add_local_python_source("clipper")' in media_block

    for image_name in ("state_image", "text_image", "speech_image"):
        assignment_index = source.index(f"{image_name} =")
        next_section = source.find("\n\n", assignment_index)
        block = source[assignment_index : next_section if next_section >= 0 else None]
        assert '.add_local_python_source("clipper")' in block, image_name

    for function_name in ("deployment_identity", "credential_smoke"):
        assert "image=state_image" in _function_decorator(source, function_name), function_name

    assert '.add_local_python_source("clipper")' in _function_decorator(source, "hf_access_smoke")
