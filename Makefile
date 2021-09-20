ENV_NAME := deepbgc-dev
SHELL := /bin/bash
CONDA_ACTIVATE = eval "$$(conda shell.bash hook)" && conda activate $(ENV_NAME)
.PHONY: install release dist test
 
env:
	conda create -n $(ENV_NAME) -c bioconda python=3.7 hmmer prodigal
	$(CONDA_ACTIVATE); pip install numpy; pip install . pytest pytest-mock hmmlearn

download:
	$(CONDA_ACTIVATE); deepbgc download

bioconda-install:
ifndef VERSION
	$(error "Usage: make conda-env VERSION=0.1.9")
endif
	conda create -n deepbgc-$(VERSION) python==3.7 deepbgc==$(VERSION)

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
	$(CONDA_ACTIVATE); pytest test

local-test:
	mkdir -p work
	rm -rf work/BGC0000015
	deepbgc pipeline test/data/BGC0000015.fa --output work/BGC0000015
	deepbgc pipeline test/data/labelled.gbk --output work/labelled
