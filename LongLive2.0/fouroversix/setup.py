import functools
import os
import platform
import subprocess
import sys
import urllib.request
import warnings
from pathlib import Path
from typing import Any

import torch
from packaging.version import Version, parse
from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

BASE_WHEEL_URL = "https://github.com/mit-han-lab/fouroversix/releases/download"
PACKAGE_NAME = "fouroversix"
PACKAGE_VERSION = "1.1.0"

CUTLASS_DEBUG = os.getenv("CUTLASS_DEBUG", "0") == "1"
FORCE_BUILD = os.getenv("FORCE_BUILD", "0") == "1"
FORCE_CXX11_ABI = os.getenv("FORCE_CXX11_ABI", "0") == "1"
SKIP_CUDA_BUILD = os.getenv("SKIP_CUDA_BUILD", "0") == "1"


@functools.cache
def get_cuda_archs() -> list[str]:
    return os.getenv("CUDA_ARCHS", "100;103;110;120").split(";")


def get_cuda_bare_metal_version() -> Version | None:
    if CUDA_HOME is None:
        warnings.warn(
            "nvcc was not found. Are you sure your environment has nvcc available? If "
            "you're installing within a container from "
            "https://hub.docker.com/r/pytorch/pytorch, only images with 'devel' in "
            "their name will provide nvcc.",
            stacklevel=1,
        )
        return None

    raw_output = subprocess.check_output(
        [CUDA_HOME + "/bin/nvcc", "-V"],
        universal_newlines=True,
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    return parse(output[release_idx].split(",")[0])


def get_cuda_gencodes() -> list[str]:
    """
    Add -gencode flags based on nvcc capabilities.

    Uses the following rules:
      - sm_100/120 on CUDA >= 12.8
      - Use 100f on CUDA >= 12.9 (Blackwell family-specific)
      - Map requested 110 -> 101 if CUDA < 13.0 (Thor rename)
      - Embed PTX for newest arch for forward compatibility
    """

    archs = set(get_cuda_archs())
    cuda_version = get_cuda_bare_metal_version()
    cc_flags = []

    # Blackwell requires >= 12.8
    if cuda_version is not None and cuda_version >= Version("12.8"):
        if "100" in archs:
            cc_flags += ["-gencode", "arch=compute_100a,code=sm_100a"]

        if "103" in archs:
            cc_flags += ["-gencode", "arch=compute_103a,code=sm_103a"]

        # Thor rename: 12.9 uses sm_101; 13.0+ uses sm_110
        if "110" in archs:
            if cuda_version >= Version("13.0"):
                cc_flags += ["-gencode", "arch=compute_110f,code=sm_110"]
            elif cuda_version >= Version("12.9"):
                # Provide Thor support for CUDA 12.9 via sm_101
                cc_flags += ["-gencode", "arch=compute_101f,code=sm_101"]
            # else: no Thor support in older toolkits

        if "120" in archs:
            # sm_120 is supported in CUDA 12.8/12.9+ toolkits
            if cuda_version >= Version("12.9"):
                cc_flags += ["-gencode", "arch=compute_120f,code=sm_120"]
            else:
                cc_flags += ["-gencode", "arch=compute_120a,code=sm_120a"]

    return cc_flags


def get_platform() -> str:
    if sys.platform.startswith("linux"):
        return f"linux_{platform.uname().machine}"
    if sys.platform == "darwin":
        mac_version = ".".join(platform.mac_ver()[0].split(".")[:2])
        return f"macosx_{mac_version}_x86_64"
    if sys.platform == "win32":
        return "win_amd64"

    msg = f"Unsupported platform: {sys.platform}"
    raise ValueError(msg)


def get_wheel_url() -> tuple[str, str]:
    torch_version_raw = parse(torch.__version__)
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_name = get_platform()
    torch_version = f"{torch_version_raw.major}.{torch_version_raw.minor}"
    cxx11_abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()  # noqa: SLF001

    # We only compile for CUDA 12.8 to save CI time. Minor versions should be
    # compatible.
    torch_cuda_version = parse("12.8")
    cuda_version = f"cu{torch_cuda_version.major}"

    wheel_filename = (
        f"{PACKAGE_NAME}-{PACKAGE_VERSION}+{cuda_version}torch{torch_version}"
        f"cxx11abi{cxx11_abi}-{python_version}-{python_version}-{platform_name}.whl"
    )

    return f"{BASE_WHEEL_URL}/v{PACKAGE_VERSION}/{wheel_filename}", wheel_filename


class CachedWheelsCommand(bdist_wheel):
    """
    Custom bdist wheel command that checks for pre-built wheels on GitHub Releases.

    The CachedWheelsCommand plugs into the default bdist wheel, which is ran by pip
    when it cannot find an existing wheel (which is currently the case for all
    fouroversix installs). We use the environment parameters to detect whether there is
    already a pre-built version of a compatible wheel available and short-circuits the
    standard full build pipeline.

    Credit: https://github.com/Dao-AILab/flash-attention/blob/main/setup.py
    """

    def run(self) -> None:
        """Run the command."""

        if FORCE_BUILD:
            return super().run()

        wheel_url, wheel_filename = get_wheel_url()
        print(f"Guessing wheel URL: {wheel_url}")

        try:
            urllib.request.urlretrieve(wheel_url, wheel_filename)  # noqa: S310

            # Make the archive
            # Lifted from the root wheel processing command
            # https://github.com/pypa/wheel/blob/cf71108ff9f6ffc36978069acb28824b44ae028e/src/wheel/bdist_wheel.py#LL381C9-L381C85
            if not Path(self.dist_dir).exists():
                Path(self.dist_dir).mkdir(parents=True, exist_ok=True)

            impl_tag, abi_tag, plat_tag = self.get_tag()
            archive_basename = f"{self.wheel_dist_name}-{impl_tag}-{abi_tag}-{plat_tag}"

            wheel_path = Path(self.dist_dir) / (archive_basename + ".whl")
            print(f"Raw wheel path: {wheel_path}")
            Path(wheel_filename).rename(wheel_path)
        except (urllib.error.HTTPError, urllib.error.URLError):
            print("Precompiled wheel not found. Building from source...")
            # If the wheel could not be downloaded, build from source
            super().run()


class NinjaBuildExtension(BuildExtension):
    """
    Custom build extension that tells Ninja how many jobs to run.

    Credit: https://github.com/Dao-AILab/flash-attention/blob/main/setup.py
    """

    def __init__(self, *args: list[Any], **kwargs: dict[str, Any]) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            try:
                import psutil

                # calculate the maximum allowed NUM_JOBS based on cores
                max_num_jobs_cores = max(1, os.cpu_count() // 2)

                # calculate the maximum allowed NUM_JOBS based on free memory
                free_memory_gb = psutil.virtual_memory().available / (
                    1024**3
                )  # free memory in GB
                max_num_jobs_memory = int(
                    free_memory_gb / 9,
                )  # each JOB peak memory cost is ~8-9GB when threads = 4

                # pick lower value of jobs based on cores vs memory metric to minimize
                # oom and swap usage during compilation
                max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
                os.environ["MAX_JOBS"] = str(max_jobs)
            except ImportError:
                warnings.warn(
                    "psutil not found, install psutil and ninja to get better build "
                    "performance",
                    stacklevel=1,
                )

        super().__init__(*args, **kwargs)


if SKIP_CUDA_BUILD:
    warnings.warn(
        "SKIP_CUDA_BUILD is set to 1, installing fouroversix without quantization and "
        "matmul kernels",
        stacklevel=1,
    )

    ext_modules = None
else:
    if Path(".git").exists():
        subprocess.run(
            [
                "git",
                "submodule",
                "update",
                "--init",
                "third_party/cutlass",
            ],
            check=True,
        )
    elif not Path("third_party/cutlass").exists():
        msg = (
            "third_party/cutlass is missing, please use source distribution or git "
            "clone"
        )
        raise RuntimeError(msg)

    # The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True  # noqa: SLF001

    setup_dir = Path(__file__).parent
    kernels_dir = setup_dir / "src" / "fouroversix" / "csrc"
    sources = [
        path.relative_to(Path(__file__).parent).as_posix()
        for ext in ["**/*.cu", "**/*.cpp"]
        for path in kernels_dir.glob(ext)
    ]

    cxx_compile_args = ["-std=c++17"]
    nvcc_compile_args = [
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-Xcompiler",
        "-funroll-loops",
        "-Xcompiler",
        "-finline-functions",
        *get_cuda_gencodes(),
    ]

    if CUTLASS_DEBUG:
        nvcc_compile_args.extend(
            [
                "-O0",
                "-DCUTLASS_DEBUG_TRACE_LEVEL=3",
                "-DCUTLASS_DEBUG_ENABLE=1",
                "-g",
            ],
        )
    else:
        cxx_compile_args.extend(["-O3"])
        nvcc_compile_args.extend(["-O3", "-DNDEBUG"])

    ext_modules = [
        CUDAExtension(
            "fouroversix._C",
            sources,
            extra_compile_args={"cxx": cxx_compile_args, "nvcc": nvcc_compile_args},
            include_dirs=[
                setup_dir / "third_party/cutlass/examples/common",
                setup_dir / "third_party/cutlass/include",
                setup_dir / "third_party/cutlass/tools/util/include",
                kernels_dir / "include",
            ],
        ),
    ]

setup(
    name=PACKAGE_NAME,
    version=PACKAGE_VERSION,
    ext_modules=ext_modules,
    cmdclass={
        "bdist_wheel": CachedWheelsCommand,
        "build_ext": NinjaBuildExtension,
    },
    include_package_data=True,
)
