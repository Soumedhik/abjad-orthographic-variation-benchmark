"""Setup script for orthographic_nli package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="orthographic-nli",
    version="1.0.0",
    description="Benchmark for evaluating LLM robustness to orthographic variation in low-resource languages",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Shibam Mandal, Soumedhik Bharati, Swarup Kr Ghosh, Sayani Mondal",
    author_email="soumedhikbharati@gmail.com",
    url="https://github.com/Soumedhik/abjad-orthographic-variation-benchmark",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="nlp, multilingual, robustness, orthographic-variation, low-resource, xnli, arabic, urdu",
    project_urls={
        "Bug Reports": "https://github.com/Soumedhik/abjad-orthographic-variation-benchmark/issues",
        "Source": "https://github.com/Soumedhik/abjad-orthographic-variation-benchmark",
    },
)
