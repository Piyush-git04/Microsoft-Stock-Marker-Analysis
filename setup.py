from setuptools import setup, find_packages

setup(
    name="microsoft-stock-forecast",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.20.0",
        "matplotlib>=3.5.3",
        "numpy>=1.24.3",
        "joblib>=1.3.1",
        "pandas>=1.5.3"
    ],
    python_requires=">=3.7",
)