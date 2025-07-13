"""
Setup script for the HCWS package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hcws",
    version="0.1.0",
    author="HCWS Team",
    author_email="contact@hcws.ai",
    description="Hyper-Conceptor Weighted Steering for Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hcws-team/hcws",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.0.0",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hcws-demo=examples.shakespeare_style:main",
        ],
    },
    keywords="artificial intelligence, machine learning, language models, steering, control",
    project_urls={
        "Bug Reports": "https://github.com/hcws-team/hcws/issues",
        "Source": "https://github.com/hcws-team/hcws",
        "Documentation": "https://hcws.readthedocs.io/",
    },
) 