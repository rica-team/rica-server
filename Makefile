.PHONY: install test lint clean

install:
	pip install .[dev,pt]

test:
	pytest

lint:
	ruff check .
	black --check .

format:
	black .

clean:
	rm -rf build dist *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +
