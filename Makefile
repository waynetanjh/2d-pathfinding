setup:
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

test:
	. venv/bin/activate && pytest test_pathfinder.py -v

clean:
	rm -rf __pycache__ .pytest_cache output/ venv/
