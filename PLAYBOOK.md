# MSN Article Generator — Pipeline Playbook
### How the workflow thinks, and why each step exists

---

## The core idea

The workflow exists to solve one problem: **a writer gives you a title and a slide count, and you need to produce a fully-sourced, fact-verified, MSN-safe slideshow in minutes, without a human researcher.**

That sounds simple, but it's full of landmines. AI models hallucinate stats. Sources go stale. Perplexity refuses questions about the future. Claude goes overloaded. Article bodies drift in quality from slide 3 to slide 20. MSN has content policies that will reject articles silently.

Every step in this pipeline exists because one of those landmines went off during earlier versions and burned a writer.

---

## Phase 1 — Understanding what you were asked for

### Step 1 · Parse the form input

**What happens:** The writer fills in a form: title, category, slide count, optional source URLs, must-include items, style notes, and extra context. This step turns that raw form data into a structured object the rest of the pipeline can work with.

**Why it matters:** The rest of the pipeline makes dozens of decisions based on this data. Getting the parsing right here means nothing downstream needs to second-guess the input.

**The temporal context block** is one of the most important things this step builds. It figures out what "current season" and "last season" mean right now, per sport. NFL seasons start in September. NBA seasons start in October. Golf is calendar-year. WNBA starts in May. If you don't build this context explicitly and inject it into every AI prompt, models default to their training cutoff and produce stats from two seasons ago. The `dateAnchor` and `seasonAnchor` strings get pasted into every single AI call later in the pipeline.

**The title analysis block** dissects the headline into its promises:
- Does the title contain a number? That's a contract with the reader — exactly that many slides.
- Does the title contain a ranking word ("top", "best", "worst")? That means slides must run in reverse order so the #1 reveal is last.
- Does the title contain an emotion word ("shocking", "heartbreaking")? That emotion must be explained — who felt it, when, why — not just stated.
- Does the title imply a correlation ("mock draft fit", "how X helps Y")? Every slide needs to connect two things, not describe them separately.

These flags travel through the entire pipeline and control how the article prompt is written and how the output is validated.

**The format config block** handles the "2 slides per entity" option. If a writer picks this, every person/team/movie in the list gets two slides: an intro slide and a deep-dive slide. The entity count is halved accordingly, and continuation naming style is detected from any user context pasted in.

---

## Phase 2 — Getting raw material to write from

### Step 2 · Route on source URL

**What happens:** If the writer pasted a URL, the pipeline goes left (scrape it). If they didn't, or if the URL is on a restricted domain list, the pipeline goes right (skip to research strategy).

**Why it matters:** The URL a writer pastes is usually the source their article is based on — a league ranking, an ESPN listicle, a box office chart. That source is Tier 0 in the editorial hierarchy: its rankings, its order, its stats override everything else. If you ignore it and let the AI invent a different order, you get an article that contradicts the page you're sourcing. That's an editorial failure.

**The restricted domain list** (Twitter, NYT, The Athletic, Bloomberg, Reddit, etc.) exists because either those sites block scraping, require a login, or are paywalled. Trying to scrape them wastes time and returns garbage. Better to detect them upfront and skip to research.

---

### Step 3 · Scrape the primary source (Firecrawl)

**What happens:** Firecrawl strips the URL down to clean markdown — just the main article content, no nav, no ads, no footer, no sidebar.

**Why Firecrawl specifically:** Raw `fetch()` on modern news sites returns React shells with no content. Firecrawl runs a headless browser, waits for JS to render, then extracts only what matters. The `onlyMainContent: true` flag is critical — without it you get 2,000 words of navigation menus.

**Timeout is 60 seconds.** Some sports stats pages are slow. The retry policy (2 attempts, 2s wait) handles intermittent failures. If it fails, the pipeline continues with empty markdown — it doesn't crash. Research fills the gap.

---

### Step 4 · Scrape secondary sources (if any)

**What happens:** If the writer pasted 2 or 3 URLs, the second one gets scraped here. `onlyMainContent` is set to `false` for secondary sources because they're often used for supplementary data (stats tables, historical context) where sidebar content might actually be useful.

**Why this matters:** A writer researching "Top 25 NCAA Women's Basketball Teams" might paste the ESPN ranking as their primary source and a separate stats breakdown as secondary. The pipeline needs both to write each slide with real numbers, not just rankings.

---

### Step 5 · Analyze source alignment

**What happens:** The pipeline checks whether the scraped content actually matches what the title is asking for. It runs three checks:
1. **Keyword match rate** — how many meaningful words from the title appear in the source?
2. **Estimated item count** — does the source have a list structure with roughly the right number of items?
3. **Fact type presence** — does the source have stats (numbers with units), dates (4-digit years), and named people?

These produce an alignment score from 0 to 100. Above 60 = use the source confidently. 35–60 = use it but supplement heavily. Below 35 = treat it like no source was given.

**Why this matters:** Writers sometimes paste loosely related URLs. A writer making a slideshow about "NFL Quarterbacks in 2025" might paste a general NFL news article that has no QB rankings in it. Without this check, the pipeline would blindly treat that article as authoritative and get confused. The alignment check gives the pipeline honest self-awareness about what it's working with.

**Multi-source bonus:** Having 2+ valid sources adds 10 points to the alignment score because more sources means more coverage.

---

### Step 6 · Build research strategy (no-source path)

**What happens:** If no valid source was provided, or if alignment was poor, this step picks the best authoritative domains to search from — ESPN and NFL.com for football, basketball-reference.com for NBA, IMDB and Rotten Tomatoes for Movies & TV, and so on.

**Why this matters:** This context gets passed to Perplexity so the research step knows which domains to prioritize when it searches. Without it, Perplexity might cite aggregator blogs instead of the primary source for a stat.

---

### Step 7 · Atomize the facts

**What happens:** The scraped source content is broken into discrete, machine-readable fact units. For each list item found in the source (a player, a movie, a team), the pipeline extracts:
- Stats (yards, points, percentages, dollar amounts)
- Achievements (MVPs, championships, Pro Bowls)
- Dates (any 4-digit year)
- Exact quotes (text in quotation marks, 15–150 characters)

It also builds a **source signature list** — 6-word phrases extracted from the source — which will later be used to detect if Claude accidentally plagiarized phrasing from the source instead of rewriting it.

**The fact-only representation** is a condensed, structured text summary of everything extracted. This goes into the AI prompts later as a "here are the facts, now write in your own words" block. Separating facts from prose is the key to getting AI to write originally rather than paraphrase.

**Why this matters:** If you just dump the raw scraped markdown into Claude and say "write a slideshow," it often paraphrases the source sentence by sentence. By pre-extracting only the raw facts (numbers, dates, names) and discarding the source's phrasing, you force the model to construct new sentences. That's the difference between original content and a derivative rewrite.

---

## Phase 3 — Research

### Step 8 · Route on research complexity

**What happens:** The pipeline decides between two Perplexity calls — a deep multi-search pass or a standard single-pass — based on four signals:

| Signal | Threshold | Why it triggers deep research |
|---|---|---|
| Slide count | ≥ 15 slides | More items = thinner coverage per item = need more data |
| Must-include list | ≥ 8 items | Many mandatory items = many individual lookups needed |
| No valid user source | — | No starting material means research has to do all the work |
| Correlation title | detected | "Mock draft fit" articles need two datasets matched together |

---

### Step 9 · Perplexity research (deep or standard)

**What happens:** Perplexity is called with live web search enabled. The system prompt does four important things:

1. **Injects the temporal context** — the date anchor and season anchor strings from Step 1. This prevents the model from returning data from two seasons ago.

2. **Bans meta-commentary** — "I need clarification", "I cannot find", "Here is the research I found" — all explicitly forbidden. The model must output structured data, not conversation.

3. **Mandates a multi-search loop** — the model is instructed to search for the overall list first, then search individually for each item. A single search rarely gets deep enough data for 20 items.

4. **Enforces citation integrity** — every source URL must actually contain the fact it's cited for. The model is told never to assign a URL to a fact it didn't verify at that URL. This is the single biggest source of AI hallucination in citation-heavy tasks.

**Deep research uses `sonar`** (Perplexity's online-search model). The user prompt is more explicit about the search strategy and mandates individual lookups per item.

**Standard research also uses `sonar`** but with a slightly lighter prompt — still rigorous, just less instruction overhead for simpler topics.

**Why not just use Claude for research?** Claude's training data has a knowledge cutoff. Perplexity has live web access and returns citations. For sports stats that change every season, live search is mandatory.

---

### Step 10 · Validate the Perplexity response

**What happens:** Before moving on, the pipeline checks if Perplexity actually answered the question. It looks for two failure modes:

1. **Hard refusal** — phrases like "cannot provide", "knowledge cutoff", "future event" indicate the model explicitly refused to answer. This is an automatic retry trigger.

2. **Thin response** — fewer than 150 words means the model returned something too short to be useful. Also triggers retry.

**Soft refusals** ("not yet been announced", "fragmented data") are flagged but don't automatically trigger retry — they're just warnings passed through for the writer to see.

**Why this matters:** Perplexity sometimes refuses questions about upcoming events, future seasons, or draft picks because it interprets them as about things that haven't happened yet. The retry uses a stronger model (`sonar-pro`) with reworded prompts to get past these refusals.

---

### Step 11 · Retry with sonar-pro (if needed)

**What happens:** If the validator flagged a retry, the same research query is sent to `sonar-pro` — Perplexity's more capable model. The retry response replaces the original only if it's longer (more complete). If the retry is somehow worse, the original is kept.

**Why not always use sonar-pro?** Cost and latency. sonar handles 90% of queries fine. sonar-pro is the escalation path for genuinely hard queries.

---

### Step 12–13 · Scrape two citation URLs from Perplexity's results

**What happens:** Perplexity returns a list of citation URLs alongside its answer. The pipeline picks two URLs that aren't the primary user source and aren't on the restricted domain list, then scrapes their full content with Firecrawl.

**Why this matters:** Perplexity's answer is often a synthesis — it states facts but compresses them. Scraping the actual source pages gives the Claude prompt access to the full original text: the exact quote from the interview, the complete stat line from the box score, the full narrative from the recap. This is the difference between Claude writing "he scored 28 points" and "he scored 28 points on 11-of-17 shooting in his first game back from injury, ending a 4-game losing streak."

The scraped citation content is truncated to 600 words each before going into the Claude prompt to stay within token budget.

---

## Phase 4 — Assembling the full prompt

### Step 14 · Build the Claude prompt

**What happens:** Everything collected so far — the temporal context, the title analysis flags, the atomized facts, the Perplexity research, the scraped citation content, the writer's style note — gets assembled into a single, very long system prompt and a user prompt.

**The system prompt is structured as a rulebook**, not a request. It has named sections:

- **Absolute output rule** — Claude must produce the full article no matter what. Refusals, "I need more data" responses, and incomplete articles are explicitly listed as forbidden outputs. If data is thin, use training knowledge and mark facts with [*].
- **Source hierarchy** — Tier 0 (user source) overrides Tier 1 (Perplexity) which overrides Tier 2 (training knowledge). Tier 3 (speculation) is never allowed.
- **Originality rule** — facts from the database can be used freely, but every sentence must be written fresh. Paraphrasing is the enemy.
- **Title-body correlation** — the title's numerical promise, its emotional promise, its main angle, and its secondary angle all must be fulfilled literally in the body. If the title says "20 quarterbacks," there must be exactly 20.
- **Word count rules** — intro slide max 60 words, content slides 35–50 words. These are hard limits, not suggestions.
- **Writing voice rules** — the "spicy writer factor." The model is told to react to facts, not summarize them. Short punchy sentences. Contrast setups. Specificity over adjectives. The "would you screenshot this and send it?" test.
- **Quality consistency engine** — this is the most important section. It tells Claude to pre-plan every slide's anchor fact before writing slide 1, then run a "sliding window check" every 5 slides to make sure quality isn't decaying. Without this, Claude writes brilliant slides 2–5 and then produces thin filler for slides 15–20.
- **Punctuation bans** — no em-dashes, no semicolons, no ellipsis. These appear often in AI-generated text and flag the content as machine-written.
- **Banned AI phrases** — a list of 80+ words and constructions ("delve", "embark", "tapestry", "it is worth noting", "the rest is history") that mark content as generically AI-generated.
- **Banned content words** — MSN's content policy is strict. A list of words that cannot appear anywhere in the output.
- **MSN safety rule** — before every slide, the model is told to ask "should a 10–12 year old be reading this?"

**The user prompt** contains the actual fact database and the writing assignment. It's kept separate from the system prompt so the system prompt can be cached (Anthropic's prompt caching feature, passed via the `anthropic-beta: prompt-caching-2024-07-31` header, caches the system prompt for 5 minutes — on multi-article runs this halves the cost of Claude calls).

**Ranking order:** If the title analysis flagged this as a ranking slideshow, the user prompt explicitly instructs Claude to order slides from lowest to highest rank — the #1 item is always the last slide. This is the key engagement mechanic for ranked slideshows: you can't skip to the end if you haven't seen the whole list.

---

## Phase 5 — Generation with fallback

### Step 15 · Claude generates the article

**What happens:** The full prompt is sent to `claude-sonnet-4-20250514` with max 6,000 tokens and temperature 0.3. The system prompt uses Anthropic's prompt caching.

**Why temperature 0.3?** Low enough to keep facts stable and avoid hallucination, high enough to prevent completely robotic sentence patterns.

---

### Step 16 · Check the Claude response

**What happens:** The response is inspected for actual failure, not just imperfect output. Three failure modes are checked:
1. `type: "error"` in the response root — Claude's API error format.
2. An `error` object with type `overloaded_error` or `rate_limit_error` — Claude is overwhelmed.
3. Article text shorter than 200 characters — the response came through but contained nothing useful.

If any of these are true, `claudeFailed = true` and the pipeline escalates to Grok.

**Important distinction:** Claude sometimes returns text that has quality issues (banned phrases, wrong word count) but is technically a complete article. Those are not "failures" — they're fixed downstream in validation. A failure is only when Claude returns nothing usable at all.

---

### Step 17 · Grok fallback (if Claude failed)

**What happens:** The exact same system prompt and user prompt are sent to `grok-4.20-0309-reasoning` via the xAI API. Grok uses a different underlying model and infrastructure, so it's statistically unlikely to fail at the same moment Claude does.

**Why keep the same prompts?** The Claude prompt was carefully engineered. Grok handles it well. There's no benefit to rewriting the prompt — it adds latency and introduces a second prompt surface to maintain.

**Timeout is 5 minutes** (vs 2 minutes for Claude) because Grok's reasoning model is slower. The extra wait is worth it rather than returning an empty article.

The Grok response is normalized into the same data structure as the Claude response so all downstream steps don't care which model generated the article.

---

## Phase 6 — Validation

### Step 18 · Structural validator

**What happens:** The article is parsed into individual slides and run through a battery of rule checks:

**Hard errors (stop the writer):**
- Intro slide over 60 words
- Any slide containing unsafe/banned content words

**Warnings (flag for review):**
- Content slides under 35 or over 50 words
- Meta description over 120 characters
- Meta using AI-generated CTAs ("Discover the...", "Explore...")
- Banned phrases present in the text
- Em-dashes, semicolons, or ellipsis in slide copy
- Intro potentially spoiling list items
- Quality degradation (last-third slides average 20%+ fewer words than first-third)
- More than 2 slides starting with the same word

**Auto-fixes (applied silently):**
- Filler transition words ("moreover", "furthermore", "ultimately", "in conclusion") are stripped automatically without flagging.

**Plagiarism check:** The 6-word source signatures extracted in Step 7 are compared against the generated article. Each match adds 5 to a plagiarism score. Score above 20 triggers a warning. Score above 30 is flagged as high risk. This is a lightweight heuristic, not a full plagiarism detector — but it catches the most common failure mode of Claude lifting phrases directly from the scraped source.

---

## Phase 7 — Fact verification

### Step 19 · Extract claims for verification

**What happens:** The article text is scanned for three types of verifiable claims:
1. **Stats** — numbers attached to sports units (yards, points, touchdowns, rebounds, percentages, dollar amounts, titles)
2. **Dates** — any 4-digit year
3. **Superlatives** — sentences containing "first", "only", "most", "best", "worst", "all-time", "record"

Up to 25 unique claims are extracted with their surrounding context (40 characters before and after the claim).

**Why extract before verifying?** Sending the full article to a fact-checker and asking it to find errors is slower and less reliable than telling it exactly which claims to check. Targeted verification gets higher accuracy.

---

### Step 20 · Perplexity verifies the claims

**What happens:** The extracted claims are sent to Perplexity with live search and a strict output format:
```
CLAIM [number]: [the claim]
STATUS: VERIFIED / INCORRECT / UNVERIFIABLE
FINDING: [what was found]
SOURCE: [URL]
```

Each claim gets one of three verdicts:
- **VERIFIED** — the fact was confirmed at the cited source
- **INCORRECT** — a different value was found (e.g., the article says 4,200 yards but the record shows 3,892)
- **UNVERIFIABLE** — no authoritative source could confirm or deny it (common for obscure historical stats)

The verification score (% verified out of total checked) becomes one of four inputs to the final quality score.

---

## Phase 8 — Full audit

### Step 21 · Grok audits and corrects the article

**What happens:** The full article is sent to `grok-3-latest` with a two-part job:

**Job 1 — Independent fact verification:** Grok re-verifies every specific claim independently, without seeing the Perplexity results. This is a second opinion, not a confirmation of the first.

**Job 2 — Rule compliance audit:** Grok checks every rule from the Claude prompt rulebook:
- Word counts per slide
- Title-body correlation
- Intro quality (no list reveals, no generic openers, has a real hook)
- Punctuation bans
- Banned phrases (and removes them)
- MSN safety

**Output:** Grok returns a corrected version of the full article with violations fixed, a fact verification section per slide, a master source list, a rule compliance section, and an audit summary with counts.

**Why use Grok for the audit instead of Claude?** Two reasons:
1. A different model gives an independent read. Claude auditing its own output often misses its own patterns.
2. Grok's live search means it can verify facts in the same call it's editing the article — it doesn't need a separate research step.

**Why is the system prompt cached with `ephemeral`?** The audit system prompt is long (the full rulebook). Caching it on Grok's side (if supported) or structuring it as the first message keeps the token cost down across high-volume runs.

**Timeout is 6–8 minutes.** The audit is the most compute-intensive step. Grok needs to search the web for every flagged claim, apply corrections, and write a full structured audit report. This is expected to be slow.

---

### Step 22 · Extract audit results

**What happens:** The Grok audit response is parsed with regex to extract four sections: the corrected article text, the fact verification log, the master source list, and the audit summary statistics.

The corrected article replaces the original only if it's longer than 500 characters (a sanity check against Grok returning a truncated response). If the corrected article is shorter than 500 chars — something went wrong with the audit — the original is preserved.

The source lists from Perplexity and Grok are merged: if the same URL appears in both, its entry is updated to show "Verified by: Both" and both fact descriptions are concatenated. This gives a single master source list for the final output.

---

## Phase 9 — Final output

### Step 23 · Final assembly

**What happens:** All scores are combined into a single quality score:

| Dimension | Weight | What it measures |
|---|---|---|
| Research completeness | 20% | Did Perplexity return enough data? |
| Fact verification rate | 30% | % of claims verified by Perplexity |
| Structural validation | 25% | Word counts, banned phrases, punctuation |
| Originality (plagiarism) | 25% | How different from the source text |

The quality score is a fast editorial signal — not a final judgment. A score of 70+ is publishable with minor edits. 50–70 needs review. Below 50 flags the article for rewrite.

The assembly step also builds the full audit report text that gets appended to the corrected Google Doc.

---

### Steps 24–29 · Google Docs + Sheets output

**What happens:**
1. Two Google Docs are created: `[Title] [Corrected]` and `[Title] [Original]`
2. The corrected doc gets the final article text + the full audit report appended
3. The original doc gets the pre-Grok article text (what Claude/Grok wrote before audit corrections)
4. Both docs are set to "anyone with the link can view"
5. A row is appended to the tracking spreadsheet

**Why keep both the corrected and original docs?**

Writers sometimes disagree with Grok's corrections. Keeping the original lets them compare and choose. It also creates a paper trail — if a factual dispute comes up later, the editor can see exactly what the model originally wrote vs. what the audit changed.

**Why set to "anyone can view"?** The tracking spreadsheet links to both docs. The editorial team reviews articles from the sheet. A link that requires a specific Google account to access breaks the review workflow.

**The tracking sheet** captures: timestamp, title, category, slide count, validation status (starts as "Pending" — human sets it to Approved/Rejected), corrected doc URL, original doc URL, primary source URL, writer name, and summary comment. This is the editorial queue.

---

## The retry / fallback philosophy

The pipeline has three explicit fallback points:

| Stage | Primary | Fallback | Trigger |
|---|---|---|---|
| Research | Perplexity sonar | Perplexity sonar-pro | Refusal or < 150 words |
| Generation | Claude Sonnet | Grok 4 reasoning | Error or < 200 chars |
| Output | Google Docs | (fail loudly) | 5 retries before error |

The philosophy: **never return an empty article.** A low-quality article can be improved. An empty response wastes the writer's time and breaks their workflow. Every failure mode that could produce nothing is covered with an escalation path.

---

## What V5.1 does NOT do (and why)

**No ISR / caching between runs.** Each run is independent. Perplexity is always called fresh. This is intentional — sports stats change daily, and a cached research result from yesterday's run could produce incorrect stats in today's article.

**No image generation.** The workflow produces text only. Image selection is left to the writer (MSN's editorial system handles image sourcing separately).

**No automatic publishing.** The output is a Google Doc that a human reviews, approves, and publishes. The pipeline's job is to get to "review-ready" — not "publish-ready." The quality score and audit report are tools for the reviewer, not a green-light signal.

**No parallel Perplexity + Claude calls.** These run sequentially on purpose. You need the research before you can build the prompt. You need the prompt before you can generate. There is no step where both can run simultaneously without the other's output — the dependency chain is linear.

---

## Tuning guide

| You want to... | Change this |
|---|---|
| Improve article quality for long slideshows | Lower the "deep research" threshold (currently ≥ 15 slides) |
| Reduce Perplexity cost | Raise the "deep research" threshold |
| Get stricter MSN compliance | Add words to the banned content list |
| Allow longer intros | Change the 60-word limit in the validator and the system prompt |
| Change the AI voice | Edit the "Writing Voice" and "Spicy Writer Factor" sections of the system prompt |
| Add a new sport category | Add its season logic to the `sportSeasons` map in Step 1 |
| Add a new source domain restriction | Add it to the `restrictedDomains` array |
| Change the quality score weights | Edit the four multipliers in Step 23 |
