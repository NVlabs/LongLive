from __future__ import annotations

import configparser
import os
import shutil
import subprocess
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import modal
import tomllib

if TYPE_CHECKING:
    from collections.abc import Callable

FOUROVERSIX_CACHE_PATH = Path("/fouroversix")
FOUROVERSIX_INSTALL_PATH = Path("/root/fouroversix")
KERNEL_DEV_MODE = os.getenv("KERNEL_DEV_MODE", "0") == "1"

app = modal.App("fouroversix")
cache_volume = modal.Volume.from_name("fouroversix", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")
wandb_secret = modal.Secret.from_name("wandb-secret")


class Dependency(str, Enum):
    """Dependencies to add to the base image."""

    awq = "awq"
    fast_hadamard_transform = "fast_hadamard_transform"
    flame = "flame"
    flash_attention = "flash_attention"
    fouroversix = "fouroversix"
    fp_quant = "fp_quant"
    qutlass = "qutlass"
    spinquant = "spinquant"
    transformer_engine = "transformer_engine"


class Submodule(str, Enum):
    """Submodules of Four Over Six to add to the base image."""

    cutlass = "cutlass"
    flame = "flame"
    fast_hadamard_transform = "fast_hadamard_transform"
    fp_quant = "fp_quant"
    llm_awq = "llm_awq"
    qutlass = "qutlass"
    spinquant = "spinquant"

    def has_untracked_or_unstaged_changes(self) -> bool:
        """Check if the submodule has untracked or unstaged changes."""

        git_status = subprocess.run(
            [
                "git",
                "-C",
                self.get_local_path(),
                "status",
                "--porcelain",
            ],
            check=False,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            text=True,
        )

        return bool(git_status.stdout.strip())

    def get_install_path(self) -> str:
        """Get the path where this submodule will be installed in the Modal image."""
        return f"{FOUROVERSIX_INSTALL_PATH}/{self.get_local_path()}"

    def get_local_path(self) -> str:
        """Get the path of the submodule relative to the root directory."""
        return f"third_party/{self.value.replace('_', '-')}"

    def get_remote_url(self) -> str:
        """Get the remote URL of the submodule."""

        gitmodules_path = Path(__file__).parent.parent / ".gitmodules"

        if not gitmodules_path.exists():
            gitmodules_path = FOUROVERSIX_INSTALL_PATH / ".gitmodules"

        with gitmodules_path.open() as f:
            # Remove leading whitespace to make it a valid INI file
            gitmodules_contents = "\n".join(line.lstrip() for line in f.readlines())

        config = configparser.ConfigParser()
        config.read_string(gitmodules_contents)

        for section in config.sections():
            if config[section]["path"] == self.get_local_path():
                url = config[section]["url"]
                break

        if url.startswith("https://"):
            return url

        msg = f"Unsupported remote URL format: {url}"
        raise ValueError(msg)


cuda_version_to_image_tag = {
    "12.8": "nvcr.io/nvidia/cuda-dl-base:25.03-cuda12.8-devel-ubuntu24.04",
    "12.9": "nvcr.io/nvidia/cuda-dl-base:25.06-cuda12.9-devel-ubuntu24.04",
    "13.0": "nvcr.io/nvidia/cuda-dl-base:25.09-cuda13.0-devel-ubuntu24.04",
    "13.1": "nvcr.io/nvidia/cuda-dl-base:25.12-cuda13.1-devel-ubuntu24.04",
}


def add_submodule(img: modal.Image, submodule: Submodule) -> modal.Image:
    if submodule.has_untracked_or_unstaged_changes():
        # Submodule has uncommitted changes, build image with local copy
        return img.add_local_dir(
            submodule.get_local_path(),
            submodule.get_install_path(),
            copy=True,
        )

    # Submodule has no uncommitted changes, download from remote to save time
    return img.run_commands(
        f"git clone {submodule.get_remote_url()} {submodule.get_install_path()}",
    )


def install_flash_attn() -> None:
    subprocess.run(
        ["pip", "install", "flash-attn", "--no-build-isolation"],
        check=False,
    )


def install_fouroversix() -> None:
    subprocess.run(
        [
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            "-e",
            FOUROVERSIX_INSTALL_PATH.as_posix(),
        ],
        check=False,
    )


def install_fouroversix_non_editable() -> None:
    shutil.copytree(
        FOUROVERSIX_CACHE_PATH / "build",
        FOUROVERSIX_INSTALL_PATH / "build",
    )
    subprocess.run(
        ["python", "setup.py", "build_ext", "--inplace"],
        check=False,
    )
    shutil.copytree(
        FOUROVERSIX_INSTALL_PATH / "build",
        FOUROVERSIX_CACHE_PATH / "build",
        dirs_exist_ok=True,
    )
    subprocess.run(
        [
            "pip",
            "install",
            "--no-deps",
            "--no-build-isolation",
            FOUROVERSIX_INSTALL_PATH.as_posix(),
        ],
        check=False,
    )


def install_qutlass() -> None:
    subprocess.run(
        [
            "pip",
            "install",
            "--no-build-isolation",
            Submodule.qutlass.get_install_path(),
        ],
        check=False,
    )


def get_image(  # noqa: C901, PLR0912
    dependencies: list[Dependency] | None = None,
    *,
    cuda_version: str = "12.9",
    deploy: bool = False,
    extra_env: dict[str, str] | None = None,
    extra_pip_dependencies: list[str] | None = None,
    include_tests: bool = False,
    python_version: str = "3.13",
    pytorch_version: str = "2.10.0",
    run_before_copy: Callable[[modal.Image], modal.Image] | None = None,
) -> modal.Image:
    if dependencies is None:
        dependencies = [Dependency.fouroversix]

    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

    if not pyproject_path.exists():
        pyproject_path = Path(__file__).parent.parent / "fouroversix" / "pyproject.toml"

    with pyproject_path.open("rb") as f:
        pyproject_data = tomllib.load(f)

    img = (
        modal.Image.from_registry(
            cuda_version_to_image_tag[cuda_version],
            add_python=python_version,
        )
        .entrypoint([])
        .apt_install("clang", "git")
        .uv_pip_install(*pyproject_data["build-system"]["requires"], "numpy")
        .uv_pip_install(
            f"torch=={pytorch_version}",
            extra_index_url=(
                f"https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')}"
            ),
        )
    )

    for dependency in dependencies:
        if dependency == Dependency.awq:
            img = add_submodule(img, Submodule.llm_awq).run_commands(
                f"pip install --no-deps -e {Submodule.llm_awq.get_install_path()}",
            )

        if dependency == Dependency.fast_hadamard_transform:
            img = add_submodule(img, Submodule.fast_hadamard_transform).run_commands(
                f"pip install {Submodule.fast_hadamard_transform.get_install_path()} "
                "--no-build-isolation",
            )

        if dependency == Dependency.flame:
            img = (
                img.apt_install("pciutils")
                .uv_pip_install(
                    "flash-linear-attention",
                    "ninja",
                    "psutil",
                    "git+https://github.com/pytorch/torchtitan.git@0b44d4c",
                    "tyro",
                    "wheel",
                )
                .run_commands(
                    "git clone https://github.com/fla-org/flame.git "
                    f"{FOUROVERSIX_INSTALL_PATH}/third_party/flame",
                    f"pip install -e {FOUROVERSIX_INSTALL_PATH}/third_party/flame",
                )
            )

        if dependency == Dependency.flash_attention:
            img = img.run_function(
                install_flash_attn,
                cpu=64,
                memory=128 * 1024,
                gpu="B200",
            )

        if dependency == Dependency.fouroversix:
            img = (
                add_submodule(
                    img.env(
                        {"CUDA_ARCHS": "100", "FORCE_BUILD": "1", "MAX_JOBS": "32"},
                    ),
                    Submodule.cutlass,
                )
                .add_local_file(
                    "pyproject.toml",
                    f"{FOUROVERSIX_INSTALL_PATH}/pyproject.toml",
                    copy=True,
                )
                .uv_pip_install(
                    *pyproject_data["project"]["optional-dependencies"]["evals"],
                )
                .add_local_file(
                    "setup.py",
                    f"{FOUROVERSIX_INSTALL_PATH}/setup.py",
                    copy=True,
                )
                .add_local_file(
                    "src/fouroversix/__init__.py",
                    f"{FOUROVERSIX_INSTALL_PATH}/src/fouroversix/__init__.py",
                    copy=True,
                )
            )

            if KERNEL_DEV_MODE:
                img = (
                    img.add_local_file(
                        "README.md",
                        f"{FOUROVERSIX_INSTALL_PATH}/README.md",
                        copy=True,
                    )
                    .add_local_file(
                        "LICENSE.md",
                        f"{FOUROVERSIX_INSTALL_PATH}/LICENSE.md",
                        copy=True,
                    )
                    .workdir(FOUROVERSIX_INSTALL_PATH)
                )

            img = img.add_local_dir(
                "src/fouroversix/csrc",
                f"{FOUROVERSIX_INSTALL_PATH}/src/fouroversix/csrc",
                copy=True,
            )

            if not KERNEL_DEV_MODE:
                img = img.run_function(install_fouroversix, cpu=32, memory=64 * 1024)

        if dependency == Dependency.fp_quant:
            img = add_submodule(img, Submodule.fp_quant).run_commands(
                f"pip install {Submodule.fp_quant.get_install_path()}/inference_lib",
            )

        if dependency == Dependency.qutlass:
            img = (
                add_submodule(img.apt_install("cmake"), Submodule.qutlass)
                .env({"MAX_JOBS": "32"})
                .run_function(install_qutlass, gpu="B200", cpu=32, memory=64 * 1024)
            )

        if dependency == Dependency.spinquant:
            img = add_submodule(img, Submodule.spinquant)

        if dependency == Dependency.transformer_engine:
            img = img.uv_pip_install(
                "transformer_engine[pytorch]",
                extra_options="--no-build-isolation",
            )

    if extra_pip_dependencies is not None:
        img = img.uv_pip_install(*extra_pip_dependencies)

    img = img.env({"HF_HOME": FOUROVERSIX_CACHE_PATH.as_posix(), **(extra_env or {})})

    if run_before_copy is not None:
        img = run_before_copy(img)

    # Add source files after all dependencies are added so we can avoid rebuilding when
    # they change
    for dependency in dependencies:
        if dependency == Dependency.flame:
            img = (
                img.add_local_dir(
                    "third_party/flame/custom_models",
                    f"{FOUROVERSIX_INSTALL_PATH}/third_party/flame/custom_models",
                )
                .add_local_dir(
                    "scripts/train/configs",
                    f"{FOUROVERSIX_INSTALL_PATH}/scripts/train/configs",
                )
                .add_local_file(
                    "third_party/flame/train.sh",
                    f"{FOUROVERSIX_INSTALL_PATH}/third_party/flame/train.sh",
                )
            )

        if dependency == Dependency.fouroversix:
            img = img.add_local_dir(
                "src",
                f"{FOUROVERSIX_INSTALL_PATH}/src",
                copy=deploy or KERNEL_DEV_MODE,
                ignore=lambda p: p.suffix == ".so",
            ).add_local_file(
                ".gitmodules",
                f"{FOUROVERSIX_INSTALL_PATH}/.gitmodules",
                copy=deploy or KERNEL_DEV_MODE,
            )

            if KERNEL_DEV_MODE:
                img = img.run_function(
                    install_fouroversix_non_editable,
                    cpu=32,
                    memory=64 * 1024,
                    volumes={FOUROVERSIX_CACHE_PATH.as_posix(): cache_volume},
                )

    if include_tests:
        img = img.add_local_dir("tests", f"{FOUROVERSIX_INSTALL_PATH}/tests")

    return img
