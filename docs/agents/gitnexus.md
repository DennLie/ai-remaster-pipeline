# GitNexus

The OpenCode project config starts the globally installed GitNexus MCP server
from this worktree. Use `repo: "ai-remaster-pipeline"` explicitly because the
machine also indexes other repositories.

Windows full-text search requires VC++ x64 and OpenSSL 3. This workstation has
a dedicated OpenSSL 3.5 LTS runtime under
`%LOCALAPPDATA%\Programs\GitNexus-OpenSSL3\bin`, added to the user PATH and the
project MCP environment. Existing terminals need reopening to inherit the user
PATH update. On another workstation, install that prerequisite or adjust the
MCP runtime path to its dedicated OpenSSL installation. `gitnexus doctor` must
report **Full-text search: available**; after fixing a runtime, run
`gitnexus analyze --repair-fts` if the index was built without search tables.

## Refresh and verify

From the repository root:

```powershell
gitnexus status
gitnexus analyze
gitnexus query "deflicker" -r ai-remaster-pipeline
```

The workspace index follows the checked-out files, including uncommitted source.
Refresh after edits or branch switches; inspect the reported working-tree state,
not only the indexed commit. Use `gitnexus analyze --force` if freshness is
uncertain. `.gitnexusrc` keeps analysis index-only, preserving the maintained
agent instructions and skills. `.gitnexus/` is local, ignored storage.
The `api` and `postJson` browser helpers are declared as HTTP wrappers so their
call sites can participate in endpoint dependency analysis.

## Coverage and safety

- `.gitignore` and `.gitnexusignore` exclude runtime media, models, caches,
  environments, evaluation artifacts and local agent tooling.
- ARP's own `vendor/comfyui_custom_nodes/ComfyUI-ARP/` is included; other
  vendored projects are excluded. Application source, tests and workflow files
  remain eligible for indexing.
- Dynamic Python context binding, runtime-generated ComfyUI graphs and calls
  into excluded dependencies may be incomplete. Confirm findings in source and
  tests; a missing graph edge is not proof that a dependency is absent.
- Execution-flow lists are bounded by the analyzer's entry-point and traversal
  budgets. Follow symbol references for detailed analysis instead of treating
  the generated process list as exhaustive.
- Indexing and keyword search run locally. Remote embeddings and LLM-generated
  wikis are not configured. GitNexus has its own PolyForm Noncommercial licence;
  ARP's licence does not grant commercial GitNexus usage rights.
