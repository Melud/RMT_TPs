.PHONY: all run debug

MAIN := TP1.py

all:
	pytype $(MAIN)

run: all
	python $(MAIN)

debug: all
	python -m pdb -c continue $(MAIN)
