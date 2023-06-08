from setuptools import setup


# Get the required modules for this project from requirements.txt
with open('requirements.txt') as req_file:
	install_requires = req_file.read()

# Perform the necessary setup for this project based on the required modules
setup(
	name='imessaGPT',
	version='1.0',
	install_requires=install_requires,
)
