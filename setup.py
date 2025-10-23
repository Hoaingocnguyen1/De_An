"""
setup.py
Package setup for Multimodal RAG System
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="multimodal-rag",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A production-ready multimodal RAG system with advanced retrieval and synthesis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/multimodal-rag",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "black>=23.12.1",
            "flake8>=6.1.0",
            "ipython>=8.18.1",
            "jupyter>=1.0.0",
        ],
        "layoutparser": [
            "layoutparser==0.3.4",
            # Note: detectron2 installation is complex, see documentation
        ],
    },
    entry_points={
        "console_scripts": [
            "multimodal-rag=main:main",
        ],
    },
)