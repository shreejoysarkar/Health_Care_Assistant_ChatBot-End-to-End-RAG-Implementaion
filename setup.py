from setuptools import setup, find_packages

setup(
    name="medibot",
    version="0.1",
    packages=find_packages(include=["medibot", "medibot.*"]),
)