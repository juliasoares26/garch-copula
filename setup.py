from setuptools import setup, find_packages

setup(
    name="dynamic-copula-evt-portfolio",
    version="1.0.0",
    description="Modelagem de risco de portfólio B3 via EVT, Vine Copulas e LSTM",
    author="",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.11",
        "arch>=6.2",
        "statsmodels>=0.14",
        "torch>=2.1",
        "scikit-learn>=1.3",
        "joblib>=1.3",
        "numba>=0.58",
        "yfinance>=0.2",
        "pandas-datareader>=0.10",
        "matplotlib>=3.7",
        "seaborn>=0.13",
        "tqdm>=4.66",
        "pyarrow>=14.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4",
            "pytest-cov>=4.1",
            "black>=23.0",
            "isort>=5.12",
            "mypy>=1.6",
        ],
        "notebooks": [
            "jupyterlab>=4.0",
            "ipywidgets>=8.1",
            "plotly>=5.17",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
