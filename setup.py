import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="caispp",
    version="0.0.1",
    author="Zane Durante",
    author_email="zanedurante@gmail.com",
    description="High level ML library used in CAIS++ Curriculum",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/caispp",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
          'numpy>=1.13.3, <=1.14.5',
          'tensorflow',
          'Pillow', 
          'matplotlib',
          'sklearn',
      ],
)
