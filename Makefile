# change Makefile shell to bash
# SHELL := /bin/bash

# instructions for creating a virtual environment for local testing
# Line 1: if a virtual environment has not been created, create it
# Line 2: activate the virtual environment for local testing
virtualenv-devel:
	# make virtual environment if needed
	if [ ! -f env/bin/activate ]; then python3 -m venv env; fi

# instructions for installing an editable package for local testing
install-devel: virtualenv-devel
	# activate virtual environment and install
	. env/bin/activate && python3 -m pip install --editable .

# instructions for building the package
build:
	echo "build instructions not written yet"
	# python3 -m build

# instructions for building the package distribution
dist:
	echo "distribution build instructions not written yet"

# instructions for cleaning the virtual environment
clean-virtualenv-devel:
	rm -rf env/
