"""Setup configuration for Pith."""

from pathlib import Path

from setuptools import find_packages, setup

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip() for line in requirements_file.read_text().splitlines() if line.strip() and not line.startswith("#")
    ]

setup(
    name="pith",
    version="1.0.5",
    author="Pith Contributors",
    description="Personal Knowledge Server with versioned conceptual memory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://pith.run",
    packages=find_packages(exclude=["tests", "tests.*", "scripts", "mcp-wrapper"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Typing :: Typed",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "semantic": [
            "sentence-transformers>=2.2.0",
            "openai>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pith-server=app.server:main",
        ],
    },
    include_package_data=True,
    package_data={
        "app": ["*.json", "*.yaml"],
    },
    zip_safe=False,
)
