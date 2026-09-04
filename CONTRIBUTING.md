# Contributing

Thanks for taking an interest in the TEGAT pipeline. The following document is
short because there is no continuous integration pipeline attached to this
project, but a minimum set of conventions is enforced for any contribution.

## Code style

- Python 3.9+ syntax (no walrus-operator-only constructs beyond 3.9).
- `from __future__ import annotations` in every `.py` file under `src/` and
  `scripts/`.
- No absolute Windows paths inside source code; everything routes through
  `src/paths.py`.
- Type hints on every public function (private helpers can be untyped when the
  signature is self-evident).
- Maximum line length: 120 characters.

## Testing

- New modules under `src/` should ship with at least one entry in
  `tests/test_smoke.py`.
- The full smoke test suite must pass before a PR is merged::

      pytest tests/

  Expected output: `9 passed`.

## Data / Results

- Never commit derived data into `data/`; if you want to refresh the bundled
  JSON, regenerate it with `reproduce.py` and inspect the diff.
- Never overwrite `results/*.json` with lower numbers than the previous
  release; regressions trigger a separate investigation.

## Pull requests

- One logical change per PR.
- Reference the issue or paper section number.
- Include a one-line summary and a short "what / why / how verified" block.
- The PR title should follow the pattern
  `[src|docs|data|tests] <one-line>`.
