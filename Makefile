.PHONY: install release dist test
 
install:
	python setup.py install
	pip install pytest pytest-mock hmmlearn

release:
ifndef VERSION
	$(error "Usage: make release VERSION=0.1.9")
endif
	git checkout master
	git pull
	echo "__version__ = '$(VERSION)'" > deepbgc/__version__.py
	git add deepbgc/__version__.py
	git commit -m "Set version to $(VERSION)"
	git push
	make dist
	twine upload dist/deepbgc-$(VERSION)*
	git checkout develop
	git pull
	git rebase origin/master
	@echo "Create a new release version on: https://github.com/Merck/deepbgc/releases"

dist:
	python setup.py sdist bdist_wheel	

test:
	pytest test

pip-install:
	pip install --upgrade deepbgc

local-test:
	mkdir -p work
	rm -rf work/BGC0000015
	deepbgc pipeline test/data/BGC0000015.fa --output work/BGC0000015
	deepbgc pipeline test/data/labelled.gbk --output work/labelled
