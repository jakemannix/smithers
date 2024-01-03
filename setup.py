from setuptools import find_packages, setup

setup(
    name='smithers',
    packages=find_packages(where="src"),
    version='0.1.0',
    package_dir={"": "src"},
    description='Your loyal assistant',
    author='YetAnotherUseless.com',
    license='MIT',
    entry_points={
        'console_scripts': ['smithers-search=main:search_cli']
    }
)
