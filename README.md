# machine-vision-assignment-1

Compact guide to the codebase: folders, scripts, resources, and how to run them.

## Directory overview

- `answers/` — Final assignment scripts per question (`01.py` … `10.py`). Each file runs standalone and shows the output for that question.
- `code-segments/` — Small exploratory snippets used while preparing the answers (e.g., image creation/opening demos).
- `resources/` — Input assets for the exercises (test images like `blurry-dog.jpg`, `highlights_and_shadows.jpg`, `looking_out.jpg`, plus `images_for_zoom/`).
- `results/` — Per-question result folders for saved outputs/screenshots (01/ … 10/).
- `questions/` — Assignment questions.
- `pyproject.toml` / `poetry.lock` — Poetry environment definition (deps: opencv-python, matplotlib, numpy, seaborn, pandas).

## Prerequisites

- Python 3.8+
- Poetry package manager (https://python-poetry.org/)

## How to run

From the project root (Poetry env already defined):

``` py
poetry install
poetry run python answers/01.py  # or any other answer script
```

Each script opens its own plots; no extra arguments required.
