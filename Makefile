build:
	python setup.py build

install:
	python setup.py install

upload:
	twine upload dist/*

publish: build upload
