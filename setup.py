from setuptools import setup, find_packages

requires = []
with open('requirements.txt') as reqfile:
    requires = reqfile.read().splitlines()

with open('README.md', encoding='utf-8') as readmefile:
    long_description = readmefile.read()


setup(
    name='NED',
    version='0.0.3',
    description='Neural Encoding Dataset',
    url='https://github.com/gifale95/NED',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
      "Programming Language :: Python",
      "Intended Audience :: Science/Research",
      ],
    maintainer='Alessandro Gifford',
    maintainer_email='alessandro.gifford@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=requires
)
