## What does this change?

<!-- A short description, and the issue it closes if there is one. -->

## Why?

<!-- The problem this solves. Skip if it's obvious from the description. -->

## Checklist

- [ ] `pytest` passes
- [ ] `pre-commit run --all-files` passes
- [ ] New or changed behaviour has a test
- [ ] `CHANGELOG.md` has an entry under `Unreleased`
- [ ] If an optional backend was touched: no optional container is imported at
      module scope, and `tests/test_backends.py` still passes on a core-only install
- [ ] If a public docstring changed: `tox -e docs` still builds the API reference
