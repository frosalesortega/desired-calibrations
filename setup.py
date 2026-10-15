from setuptools import setup, find_packages

setup(
    name="desired-calibrations",
    version="2.0.0",
    author="F. F. Rosales-Ortega",
    author_email="",
    description="DESIRED strong-line metallicity calibrations (Rosales-Ortega et al. 2026a)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/frosalesortega/desired-calibrations",
    packages=find_packages(),
    package_data={"desired": ["data/*.json"]},
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=["numpy>=1.20"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
