## Agent skills

### Issue tracker

Issues live as GitHub issues in `DennLie/ai-remaster-pipeline`, managed with the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical roles map 1:1 to `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` at the repo root and ADRs in `docs/adr/`. See `docs/agents/domain.md`.

### Code graph

For architecture, call chains and change impact, use GitNexus with `repo: "ai-remaster-pipeline"`; check index freshness before relying on results. For indexing, coverage limits and verification, read `docs/agents/gitnexus.md`.
