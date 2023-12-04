from setuptools import setup, find_packages

setup(
    name='MlClassifier',
    version='0.1',
    packages=find_packages(),
    description='MlClassifier project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'scipy',
        'datasets', # This might need a specific version based on your requirements
        'halo'
    ],
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 3 - Alpha',
        # Add other classifiers as needed
    ]
)
