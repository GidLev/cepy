import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cepy", # Replace with your own username
    version="0.0.6",
    author="Gidon Levakov",
    author_email="gidonlevakov@gmail.com",
    description="Implementation of the connectome embedding workflow.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gidlev/connectome_embedding",
    packages=setuptools.find_packages(),
    include_package_data = True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)

