build-src:
	python setup.py sdist

build:
	python -m build

install:
	python setup.py install

upload:
	twine upload dist/*

publish: build upload

init:
	python -m pip install build

all: build