from setuptools import setup, find_packages

setup(
    name="rs",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "mujoco",
        "mujoco-mjx",
        "jax==0.4.29",
        "jaxlib @ https://storage.googleapis.com/jax-releases/cuda12/jaxlib-0.4.29+cuda12.cudnn91-cp310-cp310-manylinux2014_x86_64.whl",
    ],
    include_package_data=True,
    description="A simple JAX-based implementation of random search for locomotion tasks using MuJoCo XLA (MJX).",
    long_description=open("README.md").read(),
    url="https://github.com/thowell/rs",
    author="Taylor Howell",
    author_email="taylor.athaniel.howell@gmail.com",
    python_requires=">=3.10",
)
