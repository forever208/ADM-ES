from setuptools import setup

setup(
    name="guided-diffusion",
    py_modules=["guided_diffusion"],
    install_requires=["blobfile==1.3.0", "torch", "tqdm"],
)
