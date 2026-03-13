from pathlib import Path
from setuptools import find_packages, setup


ROOT_DIR = Path(__file__).resolve().parent
README_PATH = ROOT_DIR / "README.md"


INSTALL_REQUIRES = [
    "isaacgym",
    "rsl-rl>=1.0.2",
    "matplotlib>=3.3",
    "scipy>=1.4",
]

EXTRAS_REQUIRE = {
    "deploy": [
        "mujoco>=3.2",
        "mujoco-python-viewer>=0.1.4",
        "onnxruntime>=1.14",
        "PyYAML>=6.0",
        "glfw>=2.6",
        "pygame>=2.1",
    ],
    "tools": [
        "trimesh>=3.23",
    ],
}
EXTRAS_REQUIRE["all"] = sorted({dep for deps in EXTRAS_REQUIRE.values() for dep in deps})


setup(
    name="legged_gym",
    version="1.0.0",
    author="Nikita Rudin",
    author_email="rudinn@ethz.ch",
    license="BSD-3-Clause",
    description="Isaac Gym environments for Legged Robots",
    long_description=README_PATH.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(include=("legged_gym", "legged_gym.*")),
    include_package_data=True,
    python_requires=">=3.8,<3.11",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
