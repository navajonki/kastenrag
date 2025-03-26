"""Setup script for KastenRAG package."""

from setuptools import setup, find_packages

setup(
    name="kastenrag",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.0.181",
        "pydantic>=1.10.8",
        "fastapi>=0.100.0",
        "uvicorn>=0.20.0",
        "whisper>=1.1.10",
        "deepgram-sdk>=3.10.0",
        "pydub>=0.25.1",
        "spacy>=3.7.0",
        "nltk>=3.9.0",
        "chromadb>=0.6.0",
        "neo4j>=5.22.0",
        "py2neo>=2021.2.3",
        "openai>=1.0.0",
        "anthropic>=0.10.0",
        "replicate>=1.0.4",
        "tqdm>=4.65.0",
        "loguru>=0.7.0",
        "pyyaml>=6.0.0",
        "python-dotenv>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "kastenrag=kastenrag.app:main",
        ],
    },
    python_requires=">=3.8",
)