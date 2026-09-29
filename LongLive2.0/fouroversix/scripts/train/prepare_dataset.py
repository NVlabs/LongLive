from ..resources import FOUROVERSIX_CACHE_PATH, app, cache_volume, get_image

img = get_image(dependencies=[], extra_pip_dependencies=["datasets"])

with img.imports():
    from datasets import load_dataset


@app.function(
    image=img,
    timeout=24 * 60 * 60,
    volumes={FOUROVERSIX_CACHE_PATH: cache_volume},
)
def prepare_dataset(path: str, name: str) -> None:
    load_dataset(path, name)
