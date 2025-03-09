build-src:
	python setup.py sdist

build:
	python -m build

install:
	python setup.py install

upload:
	twine upload dist/*

pub: publish

publish: build upload

init:
	python -m pip install build

clean:
	rm -rf dist
	rm -rf build
	rm -rf torch_gamma.egg-info

all: build