# Agent Python Environment Template

> Copy this file to the root of a project as `AGENTS.md`, then replace the
> `PYTHON_ENV_PATH` value below with the Python/Conda environment for that
> project.

## Required Python Environment

Change only this block when using another Python environment:

```text
PYTHON_ENV_PATH=D:\Apps\Anaconda3\envs\env_dk_cdsm
PYTHON_EXE=%PYTHON_ENV_PATH%\python.exe
```

`PYTHON_ENV_PATH` is the only value that should normally be edited. The agent
must treat `PYTHON_EXE` as the Python executable inside that environment.

All Python commands in this project must run inside this environment.
Before running any Python command, verify that the interpreter is the one above.
Do not use global Python, system Python, user-site Python, or an unrelated
virtual environment.

Use explicit interpreter commands:

```powershell
%PYTHON_EXE% -m pytest
%PYTHON_EXE% -m pip install -r requirements.txt
%PYTHON_EXE% script.py
```

Do not use bare commands unless they are already proven to resolve to this same
environment:

```powershell
python
pip
pytest
```

## Dependency Installation Policy

The agent is allowed to install Python dependencies, but only into the required
environment above.

When installing packages, always use:

```powershell
%PYTHON_EXE% -m pip install <package>
```

Never install packages with:

```powershell
pip install <package>
python -m pip install <package>
```

unless `python` has first been verified to be exactly:

```text
%PYTHON_EXE%
```

## requirements.txt Rule

After installing, upgrading, or removing any Python dependency, immediately
update the project root `requirements.txt`.

If `requirements.txt` does not exist, create it immediately.

Preferred update command:

```powershell
%PYTHON_EXE% -m pip freeze > requirements.txt
```

If the project already uses a more specific dependency workflow, such as
`pyproject.toml`, `poetry.lock`, `uv.lock`, or `environment.yml`, update the
appropriate project dependency file as well. Still keep `requirements.txt`
present unless the user explicitly says not to.

## Verification Checklist

Before running tests, scripts, or package installs, the agent must check:

```powershell
%PYTHON_EXE% -c "import sys; print(sys.executable)"
```

The printed path must be:

```text
%PYTHON_EXE%
```

After dependency changes, the agent must check that `requirements.txt` exists
and contains the installed package.

## Agent Operating Rules

- Run tests with the required environment's Python interpreter.
- Run scripts with the required environment's Python interpreter.
- Install dependencies only through the required environment's Python interpreter.
- After dependency changes, update `requirements.txt` immediately.
- Do not modify global Python, system Python, or user-site packages.
- If the configured environment does not exist or cannot run, stop and ask the
  user before using any other Python environment.
