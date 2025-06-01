SHELL :=/bin/bash

.PHONY: clean check setup
.DEFAULT_GOAL=help
ENV_NAME = llm_web  # 改成你的项目名
PYTHON_VERSION = 3.11

check: # Ruff check
	@ruff check .
	@echo "✅ Check complete!"

fix: # Fix auto-fixable linting issues
	@ruff check app.py --fix

clean: # Clean temporary files
	@rm -rf __pycache__ .pytest_cache
	@find . -name '*.pyc' -exec rm -r {} +
	@find . -name '__pycache__' -exec rm -r {} +
	@rm -rf build dist
	@find . -name '*.egg-info' -type d -exec rm -r {} +

run: # Run the application
	@conda run -n $(ENV_NAME) streamlit run app.py

setup: # Initial project setup with conda
	@echo "Creating conda environment: $(ENV_NAME)"
	@conda create -n $(ENV_NAME) python=$(PYTHON_VERSION) -y
	@echo "Installing dependencies..."
	@conda run -n $(ENV_NAME) pip install -r requirements/requirements-dev.txt
	@conda run -n $(ENV_NAME) pip install -r requirements/requirements.txt
	@echo -e "\n✅ Done.\n🎉 Run the following commands to get started:\n\n ➡️ conda activate $(ENV_NAME)\n ➡️ make run\n"

help: # Show this help
	@egrep -h '\s#\s' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?# "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
