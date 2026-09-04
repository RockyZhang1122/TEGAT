# Security policy

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Email the co-corresponding author with a short proof-of-concept and the commit
range affected. See `paper/paper.tex` (front matter) for current contact info.

## Built-in disclosure

- The bundled dataset is a derivative of publicly released Chinese provincial
  health notices. Case identifiers are anonymized sequential IDs and exact
  birth dates are not stored.
- No passwords, API keys, or non-public URLs are hard-coded.
- The included `.gitignore` excludes `.env`, `venv/`, and Windows / macOS
  thumbnails.
- No telemetry is sent from the scripts.
