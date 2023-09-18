from setuptools import setup
from pip._internal.req import parse_requirements

install_reqs = parse_requirements('requirements.txt', session=False)
reqs = [str(ir.requirement) for ir in install_reqs]

setup(
    name="syntgenerator",
    packages=setuptools.find_packages(),
    install_requires=reqs,
    author="",
    version='0.0.2'
)