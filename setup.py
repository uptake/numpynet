import setuptools

# Required packages
dependencies = [
    "numpy",
    "visdom"
]
documentation_packages = [
    "sphinx",
    "sphinxcontrib-napoleon",
    "sphinxcontrib-programoutput"
]
testing_packages = [
    "pytest",
    "coverage"
]

setuptools.setup(
    name='numpynet',
    version='0.0.1',
    author='Brad Beechler',
    author_email='Chronocook@gmail.com',
    description='Approachable neural net implementation in pure numpy',
    url='https://github.com/UptakeOpenSource/numpynet',
    packages=setuptools.find_packages(),
    install_requires=dependencies,
    extras_require={
        'all': dependencies + documentation_packages,
        'docs': documentation_packages
    },
    test_suite="tests",
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Programming Language :: Python :: 3 :: Only'
    ]
)
