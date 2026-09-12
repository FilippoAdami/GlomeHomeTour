Here's a practical, current setup for making Claude Code both fast and cheap for app dev. This is based on Anthropic's official docs plus what's actually panning out in the community as of August 2026.

## 1. Trim what loads on every single turn

This is the highest-leverage lever because it's a tax you pay on every message, forever:

- **Keep CLAUDE.md short** — ideally under a few hundred lines, purely "things Claude can't infer from reading the code" (non-standard bash commands, code style deviations, testing preferences, repo etiquette). Cut anything Claude already does correctly without the instruction. Run `/init` to generate a starting one, then prune ruthlessly. A bloated CLAUDE.md doesn't just cost tokens — Claude starts ignoring it because important lines get lost in the noise.
- **Add a `.claudeignore`** excluding `node_modules`, build output, lockfiles, binaries, and large data/fixture files so Claude never accidentally reads them into context.
- **Move situational knowledge to skills**, not CLAUDE.md. Skills in `.claude/skills/` only load when relevant, so domain-specific stuff (API conventions, a deploy runbook, a migration playbook) belongs there instead of being injected every session.

## 2. Use subagents to keep the *main* context clean

This is the single biggest lever, per Anthropic's own docs, because context degrades quality as it fills. Subagents run in their own context window and only report a summary back:

- Delegate exploration: *"use subagents to investigate how auth handles token refresh."*
- Delegate review: after implementing, spin up a fresh-context subagent to review the diff against your plan and report only correctness gaps — it won't be biased by having just written the code.
- Define custom ones in `.claude/agents/` with restricted tools (`Read, Grep, Glob` for a reviewer, no `Write`/`Bash` needed).

## 3. Route models by task difficulty

Sonnet and Opus produce near-identical output on most day-to-day coding tasks; Opus earns its cost on multi-file bugs with indirect causes, architecture decisions, and security review. Concretely:

- Set `CLAUDE_CODE_SUBAGENT_MODEL` so subagents run on a cheaper model (e.g. Sonnet or Haiku) regardless of what your main session uses.
- Reserve Opus for the main session on genuinely hard problems; drop to Sonnet/Haiku for everything else.
- Turn off extended thinking for mechanical work (renames, typo fixes) — it burns output tokens even when the task doesn't need deliberation.

## 4. Session hygiene

- `/clear` between unrelated tasks — don't let one long session accumulate irrelevant history.
- If you've corrected Claude twice on the same thing, `/clear` and rewrite the prompt with what you learned rather than correcting a third time.
- `/compact <instructions>` when a session does need to continue, e.g. `/compact focus on the API changes`.
- `/btw` for one-off questions you don't want polluting context.
- `/cost` and `/usage` to actually track where tokens go — don't optimize blind.

## 5. Plan mode for anything non-trivial

For changes touching multiple files or unfamiliar code, use plan mode (`Shift+Tab` until you see `plan mode on`) to separate exploration/planning from implementation — this avoids Claude confidently solving the wrong problem, which is far more expensive than the planning overhead. Skip it for one-sentence-describable diffs.

## 6. Plugins worth installing

Run `/plugin` to browse the official marketplace (`claude-plugins-official`, enabled by default). Worth prioritizing for app dev:

- **Code intelligence (LSP) plugin for your language** — gives Claude real jump-to-definition, find-references, and live type-error diagnostics after edits, instead of grepping text. This meaningfully cuts wasted exploration tokens and catches Claude's own mistakes immediately rather than several turns later. You need the language server binary installed separately; the plugin just wires it up.
- **GitHub integration plugin** (or just the `gh` CLI, which Claude already knows how to drive) — far more token-efficient than unauthenticated API calls for issues/PRs.
- Community dev-workflow plugins (e.g. `obra/superpowers`) if you want opinionated TDD/debugging skills bundled in — worth a look, but treat third-party plugins like dependencies: they can run code with your privileges.

## 7. Hooks for anything that must happen with zero exceptions

CLAUDE.md instructions are advisory; hooks are deterministic. Have Claude write you a hook like *"run eslint after every file edit"* or *"block writes to the migrations folder"* — this avoids relying on the model remembering a rule every single turn.

---

Rough expectation from teams applying this consistently: 40–60%+ reduction in token usage with no quality loss, since most of the waste is habit (stale context, unscoped exploration, wrong model for the job) rather than something you're giving up.

One honest caveat: subagents aren't automatically cheaper for tiny tasks — spinning one up for a simple shell command can cost more than just doing it inline, because of the added tool-definition and round-trip overhead. Use them for genuinely context-heavy work (research, multi-file review), not everything.