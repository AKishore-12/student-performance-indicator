from setuptools import find_packages, setup

HYPEN_E_DOT = "-e ."

def get_requirement(file_path):
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name="mlproject",
    version="0.0.1",
    author="Kishore",
    author_email="kishore007008ya@gmail.com",
    packages=find_packages(),
    install_requires=get_requirement("requirements.txt")
)