# setup.py

from setuptools import setup, find_packages

setup(
    name="amharic_telegram_ecommerce_ner",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
