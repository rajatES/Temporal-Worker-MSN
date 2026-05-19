# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the worker (connects to Temporal server, processes activities)
npm run worker         # ts-node src/worker.ts

# Trigger a workflow run (edit the exampleInput in src/client.ts first)
npm run trigger        # ts-node src/client.ts

# Build TypeScript
npm run build          # tsc --build

# Inspect a running or completed workflow
ts-node src/inspect-run.ts <workflowId>
ts-node src/list-runs.ts
ts-node src/fetch-result.ts <workflowId>
ts-node terminate-workflow.ts <workflowId>
ts-node terminate-all-workflows.ts
```

The worker requires a running Temporal server at `localhost:7233`. The Temporal UI is at `http://localhost:8233`.

## Architecture

This is a **Temporal.io workflow** that generates MSN slideshow articles end-to-end. A writer submits a `FormInput` (title, category, slide count, optional source URLs), and the workflow returns a `WorkflowResult` with structured slides ready for the MSN editorial system.

### Entry points

- **`src/worker.ts`** — registers all activities and the workflow, connects to the `msn-article-generator` task queue. Must be running before any workflow is triggered.
- **`src/client.ts`** — triggers a single workflow run with a hardcoded `FormInput`. Edit `exampleInput` to change what runs. Polls for result and prints quality metrics.

### Data flow

The pipeline is a **linear chain** — each activity receives the entire accumulated data object from the previous stage and spreads it forward (similar to n8n). The types in `src/types.ts` document this precisely: `PreparedData → SourcedData → AtomizedData → ResearchedData → MergedData → PromptData → GeneratedData → ValidatedData → ClaimedData → VerifiedData → AuditedData → FinalOutput`.

Do not call activities in parallel — the dependency chain is strictly sequential.

### The 12-stage pipeline (in order)

| Stage | Activity | What it does |
|---|---|---|
| 1 | `prepareInputAndAnalyze` | Parses form input; builds temporal context (sport-aware season anchors), title analysis flags (isRanking, requiresCorrelation, etc.), format config |
| 2 | `firecrawlScrape` + `analyzeSourceAlignment` or `buildResearchStrategy` | Scrapes user-supplied URLs via Firecrawl; scores alignment 0–100. <40 triggers a human review pause |
| 3 | `atomizeFacts` | Extracts discrete fact units (stats, dates, quotes) per list item; builds source signature phrases for plagiarism detection |
| 4 | `perplexityDeepResearch` / `perplexityStandardResearch` | Live web research via Perplexity sonar. Deep path triggers when slide count ≥15, mustInclude ≥8 items, no user source, or correlation title |
| 5 | `validateRetry` + optional `perplexityRetryResearch` | Detects refusals or thin responses; escalates to sonar-pro if needed |
| 6 | `firecrawlScrape` ×2 | Scrapes 2 citation URLs from Perplexity results for deeper source content |
| 7 | `buildClaudePrompt` | Assembles the full system + user prompt; system prompt is structured for Anthropic prompt caching |
| 8 | `generateWithClaude` → `checkClaudeResponse` → `generateWithGrok` (fallback) | Calls Claude Sonnet; falls back to Grok 4 reasoning on API error or <200 char response |
| 9 | `validateStructure` | Checks word counts, banned phrases, punctuation, MSN safety; applies auto-fixes silently |
| 10 | `extractClaims` + `perplexityVerifyClaims` | Extracts up to 25 verifiable claims; Perplexity live-checks each one |
| 11 | `grokAuditAndVerify` | Grok independently fact-checks and rule-audits the article; returns corrected text if violations found |
| 12 | `finalAssembly` | Computes quality score (research 20% + verification 30% + structural 25% + originality 25%); writes Google Doc + tracking sheet row |

### Human-in-the-loop

The workflow pauses at four decision points using `condition()` with a 30-minute timeout (auto-continues to `continue` if no response):
- Source alignment score <40
- Research word count too thin after retry
- Hard structural errors after auto-fix
- More than 2 fact errors from Perplexity
- Quality score <55

Decisions are sent via the `humanDecision` signal. Progress is readable via the `getProgress` query. Both are registered at module level before any `await` — this is required by Temporal's determinism rules.

### Activity timeouts

Most activities: `startToCloseTimeout: 3 minutes`. The Grok audit and Grok fallback generation use a separate proxy with `8 minutes` because Grok's reasoning model is slow. Don't merge these proxies.

### Key external services

| Service | Used for | Env var |
|---|---|---|
| Firecrawl | Headless scraping of source URLs and citations | `FIRECRAWL_API_KEY` |
| Perplexity sonar / sonar-pro | Live research and fact verification | `PERPLEXITY_API_KEY` |
| Anthropic Claude Sonnet | Article generation | `ANTHROPIC_KEY` |
| xAI Grok | Claude fallback + audit/fact-check | `GROK_API_KEY` |
| Google APIs | Create Docs + append to tracking Sheet | `GOOGLE_SERVICE_ACCOUNT_KEY_FILE`, `GOOGLE_SHEET_ID` |

Google auth uses a service account key file (JSON), not OAuth. The path is set via `GOOGLE_SERVICE_ACCOUNT_KEY_FILE`.

### Temporal determinism rules

Workflow code (`src/workflows.ts`) must be deterministic — no `Date.now()`, no `Math.random()`, no direct I/O. All side effects happen in activities. Signal and query handlers must be registered with `setHandler` before the first `await`. The `STAGE_DEFS` constant is defined in `types.ts` (not `workflows.ts`) so it can be imported by both the workflow and non-workflow code without bundling issues.

### Adding a new sport's season logic

Edit the `sportSeasons` map in `prepareInputAndAnalyze` (`src/activities.ts`). Key is the uppercase sport name as it appears after stripping `"Sports - "` from the category string.
