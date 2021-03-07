pkg-build:
	python setup.py sdist bdist_wheel
check:
	twine check dist/*
publish:
	twine upload dist/*