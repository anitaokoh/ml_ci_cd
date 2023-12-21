install_poetry:
	curl -sSL https://install.python-poetry.org | python -

install_dependencies:
	poetry install

test:
	poetry run pytest -vv --cov=tests

deploy:
	poetry run python -m src.model_building
	mkdir -p model
	mv trained_model.pkl model/
