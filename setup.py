import setuptools

with open("README.md", "r") as fh:
    readme = fh.read()

setuptools.setup(
    name="odysseus",
    version="0.0.1",
    author="Corentin Herbert",
    author_email="corentin.herbert@ens-lyon.fr",
    description="A framework to design and run fluid flow simulations based on Dedalus",
    long_description=readme,
    packages=["odysseus"],
    install_requires=["numpy", "matplotlib", "dedalus"],
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
