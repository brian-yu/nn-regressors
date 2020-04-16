import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nn-regressors",
    version="0.0.1",
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