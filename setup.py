import sys,os
from setuptools import setup, find_packages
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path

def get_version() -> str:
    # https://packaging.python.org/guides/single-sourcing-package-version/
    init = open(os.path.join("joyrl", "__init__.py"), "r").read().split()
    return init[init.index("__version__") + 2][1:-1]


def get_install_requires() -> str:
    return [
        "gym==0.25.2",
        "pyyaml==6.0",
        "matplotlib==3.5.3",
        "seaborn==0.12.1",
        "dill==0.3.5.1",
        "argparse==1.4.0",
        "pandas==1.3.5",
        "pyglet==1.5.26",
        "importlib-metadata<5.0",
        "setuptools==65.2.0"
    ]


def get_extras_require() -> str:
    req = {
        "atari": ["atari_py", "opencv-python"],
        "mujoco": ["mujoco_py"],
        "pybullet": ["pybullet"],
    }
    return req

setup(
    name="joyrl",
    version=get_version(),
    description="A Library for Deep Reinforcement Learning",
    long_description=open(f"{curr_path}/README.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/datawhalechina/joyrl",
    author="johnjim0816",
    author_email="johnjim0816@gmail.com",
    license="MIT",
    python_requires=">=3.7",
    keywords="reinforcement learning platform pytorch",
    packages=find_packages(
        exclude=["test", "test.*", "examples", "examples.*", "docs", "docs.*"]
    ),
    platforms = "any",
    install_requires=get_install_requires(),
    extras_require=get_extras_require(),
)