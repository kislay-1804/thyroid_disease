# This is the Setup File for the Project: Thyroid Disease Detection
# Author: Anirban Majumder
# Date: 17-08-2024         [Date Format: dd-mm-yyyy]

from setuptools import setup, find_packages

setup(
    name="Thyroid Disease Detection",      
    version="0.1.0", 
    author="Anirban Majumder",
    author_email="animouani123@gmail.com",
    description=open("README.md").read(),
    description_content_type="text/markdown",
    url="https://github.com/ImAni07/Thyroid-Disease-Detection-ML-Project.git",
    packages=find_packages(),
    
    install_requires=[
        "numpy", 
        "pandas",
        "flask",
        "pymongo",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "xgboost",
        "requests",
        "flask_cors",
        "imblearn",
        "gunicorn"
    ],
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    
    python_requires=">=3.7"
)
