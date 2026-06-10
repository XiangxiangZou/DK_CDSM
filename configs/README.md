# Experiment Configuration

`configs/` contains stable, reviewable defaults. Command-line arguments may
override these values for a specific run, but reusable defaults should not be
copied into individual experiment scripts.

Current configuration group:

- `deployment/experiment_common.json`: state, control, cable order, and timing.
- `deployment/models.json`: model capability declarations.
- `deployment/tracking_reference.json`: default joint tracking reference.
- `project/`: archived environment, dependency, build, editor, and agent files
  from the former repository root.
