from setuptools import find_packages, setup

setup(
    name="polish-bankruptcy-mlops",
    version="1.0.0",
    author="Boubacar Aliou Traore",
    author_email="tboubacaraliou@gmail.com",
    description="MLOps system for corporate bankruptcy prediction using Polish companies data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial Services",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.1.0",
        "scikit-learn>=1.3.0",
        "mlflow>=2.8.0",
        "fastapi>=0.104.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.11.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
        ],
    },
)
