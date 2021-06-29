install:
	python setup.py install --user

tests:
	python tests.py

clean:
	rm -r build dist *.egg-info *~ odysseus/*~
