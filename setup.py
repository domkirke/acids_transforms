import setuptools
import subprocess

with open("README.md", "r") as readme:
    readme = readme.read()

with open("requirements.txt", "r") as requirements:
    requirements = requirements.read()

setuptools.setup(
    name="acids_transforms",
    version="0.1",
    author="Axel CHEMLA--ROMEU-SANTOS",
    author_email="chemla@ircam.fr",
    description="A bunch of scriptable audio transforms based on the torchaudio backend",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements.split("\n"),
    python_requires='>=3.7',
)