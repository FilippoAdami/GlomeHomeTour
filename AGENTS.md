# Agent Repository Instructions

Project-specific agent's context and working files are maintained under
`agents_files/`.

Before substantial work, read the relevant files in `agents_files/`, starting
with `agents_files/PROJECT.md` and `agents_files/COMMANDS.md`.

Treat `agents_files/` as agents' project-local working memory and engineering
workspace.

## Repository integration

Do not reorganize or restructure the existing repository merely to fit the agent's
conventions.

Respect the project's existing architecture, naming, tooling, formatting,
testing, and documentation conventions.

Project source code, tests, and normal documentation remain in their existing
locations. Do not move them into `agents_files/`.

## Maintaining Agent's context

The agentic system may create and maintain files inside `agents_files/` when doing so helps
future work.

Keep these files concise and factual.

Update persistent project context when substantial work reveals durable
information that would help future sessions.

Do not store transient reasoning, verbose session logs, duplicated source code,
or information easily rediscovered from the repository.

## Task workspace

For large tasks, agents may create a task file under:

`agents_files/tasks/`

Use task files for durable state that helps long-running or multi-session work:
goals, acceptance criteria, phases, completed work, important discoveries,
verification state, and remaining work.

Keep task state concise and update it rather than accumulating chronological
logs.

## Research

Store durable project-relevant research under:

`agents_files/research/`

Only preserve research that is likely to be useful again. Prefer links,
versions, conclusions, compatibility notes, and concise evidence over copied
web content.

## Verification artifacts

Human-inspectable outputs generated specifically for verification should
normally go under:

`agents_files/artifacts/<task-name>/`

unless the repository already defines a more appropriate location.

## Benchmarks

Codex-specific benchmark results and comparisons may be stored under:

`agents_files/benchmarks/`

Do not mix benchmark outputs with production source code.

## Safety

Do not store secrets, credentials, tokens, private keys, or `.env` contents in
`agents_files/`.

Do not push to Git remotes.

The existing repository remains authoritative. Files under `agents_files/`
describe and support the repository; they do not override reality in the
source code, build system, tests, or authoritative project documentation.
