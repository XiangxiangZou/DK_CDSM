# Experiment Configuration

`configs/` contains stable, reviewable defaults. Command-line arguments may
override these values for a specific run, but reusable defaults should not be
copied into individual experiment scripts.

Current configuration group:

- `deployment/experiment_common.json`: state, control, cable order, and timing.
- `deployment/models.json`: model capability declarations.
- `deployment/tracking_reference.json`: default joint tracking reference.
- `project/`: archived editor and agent templates from the former repository
  root. The canonical build, pytest, and dependency files are
  `/pyproject.toml` and `/requirements.txt`.
