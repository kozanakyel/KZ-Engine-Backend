from setuptools import setup

with open('../LICENSE') as f:
    license_text = f.read()

setup(
    name="KZ_project",
    version="0.1",
    packages=["KZ_project"],
    author="Uğur (Kozan) Akyel",
    author_email="kozanakyel@gmail.com",
    license=license_text
)
