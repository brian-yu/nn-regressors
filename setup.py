import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nn-regressors", # Replace with your own username
    version="0.0.1",
    author="Brian Yu",
    author_email="bry4xm@virginia.edu",
    description="Regressors that predict NN layer and CPU usage.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brian-yu/nn-regressors",
    packages=['nn_regressors'],
    package_dir={'nn_regressors': 'nn_regressors/'},
    package_data={'nn_regressors': ['./*.joblib', './*_benchmark.txt']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
    install_required=[
        'scikit-learn',
        'numpy',
        'tensorflow',
        'matplotlib',
        'pandas',
        'joblib',
    ],
)