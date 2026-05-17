# Public Repository Layout

The original working directory contained:

- application code
- training utilities
- local datasets
- experiment outputs
- model weights
- SQLite records
- raw and processed media

For GitHub publishing, only the reusable source code and one representative detection screenshot were kept. Local-only artifacts remain excluded through `.gitignore`.

## File Mapping

| Public repository path | Source purpose |
| --- | --- |
| `main.py` | simple application entry point |
| `ui/main_ui.py` | PyQt5 desktop interface and inference workflow |
| `src/*.py` | dataset, training, extraction, and testing utilities |
| `src/yolov8-cbam.yaml` | model architecture experiment config |
| `screenshots/detection_example.jpg` | representative processed output image |
