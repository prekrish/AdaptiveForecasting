from setuptools import setup, find_packages

setup(
    name="adaptiveforecast",
    version="0.1.0",
    description="A user-friendly interface for time series forecasting using sktime",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/adaptiveforecast",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "matplotlib>=3.1.0",
        "sktime>=0.8.0",
        "scikit-learn>=0.24.0",
    ],
    extras_require={
        "mlflow": ["mlflow>=1.0.0"],
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "flake8>=3.9.0",
            "sphinx>=4.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.7",
)