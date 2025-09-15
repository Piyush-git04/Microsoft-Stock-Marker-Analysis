from setuptools import setup, find_packages

setup(
    name="microsoft-stock-forecast",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "matplotlib",
        "numpy",
        "joblib",
        "pandas"
    ],
    python_requires=">=3.7",
)