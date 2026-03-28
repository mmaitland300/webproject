# Contributing

This repo uses a lightweight workflow. Keep changes small, reviewable, and consistent.

## Commit Message Convention

Use this format:

`type(scope): summary`

Examples:

- `feat(contact): add reply status badges`
- `fix(ci): decouple build from migrate deploy`
- `docs(readme): clarify prisma env placeholders`

Rules:

- Use lowercase `type`.
- Keep `summary` concise and imperative.
- Add `scope` when a specific area is clear (`ci`, `admin`, `projects`, `docs`, etc.).
- Recommended types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`.
- Avoid generic subjects like `update`, `misc`, or `merge: ...`.

## Merge Policy

For `master`, prefer one policy and apply it consistently:

- Default: **Squash merge** PRs with an edited final title that follows the commit format.
- Keep PRs focused on one concern.
- Avoid mixing merge styles (`Merge pull request...`, `Merge branch...`, custom `merge:` commits) in normal flow.
- If a direct merge commit is required, use a clear conventional title.

## Author Identity Checklist

Before opening a PR or committing:

1. Verify identity:
   - `git config user.name`
   - `git config user.email`
2. Confirm it matches your chosen project identity (pick one and keep it stable).
3. If signing in via another machine/account, re-check before first commit.
4. After commit, confirm author on latest commit:
   - `git log -1 --format='%h %an <%ae> %s'`

## Text and Encoding Guardrails

- Use ASCII punctuation in first-party docs/UI copy when possible.
- Avoid smart quotes/em dashes/bullets that can render as mojibake in some terminals.
- If copying text from external sources, quickly scan for encoding artifacts before commit.
