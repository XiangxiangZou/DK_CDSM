# Automated Tests

Tests in this directory are non-interactive and suitable for repeated local or
CI execution. Historical scripts named `test_*.py` were interactive validation
programs and now live under `archive/diagnostics/`.

Run:

```powershell
& $env:PYTHON_EXE -m pytest
```
