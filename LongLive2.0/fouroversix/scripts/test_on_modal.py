from __future__ import annotations

from pathlib import Path

from .resources import FOUROVERSIX_INSTALL_PATH, Dependency, app, get_image

img = get_image(
    dependencies=[Dependency.transformer_engine, Dependency.fouroversix],
    include_tests=True,
)

with img.imports():
    import pytest


@app.function(image=img, cpu=4, memory=8 * 1024, gpu="B200", timeout=30 * 60)
def run_tests(*args: list[str]) -> None:
    """Run tests on a B200 on Modal."""

    args = list(args)
    tests_path = (Path(FOUROVERSIX_INSTALL_PATH) / "tests").as_posix()

    if len(args) == 0:
        args = [tests_path]
    elif "tests" in args[0] or "test_" in args[0]:
        args[0] = (Path(FOUROVERSIX_INSTALL_PATH) / args[0]).as_posix()
    else:
        args = [tests_path, *args]

    pytest.main(args)
