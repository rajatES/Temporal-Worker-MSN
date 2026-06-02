import {
  proxyActivities,
  defineQuery,
  defineSignal,
  setHandler,
  condition,
} from '@temporalio/workflow';
import type * as activities from './activities';
import type {
  FormInput,
  WorkflowResult,
  WorkflowProgress,
  Stage,
  HumanReviewRequest,
  HumanDecision,
} from './types';
import { STAGE_DEFS, STAGE_DEFS_SUBJECTIVE } from './types';

// ── Activity proxies ──────────────────────────────────────────────────────────

const {
  prepareInputAndAnalyze,
  firecrawlScrape,
  analyzeSourceAlignment,
  buildResearchStrategy,
  atomizeFacts,
  perplexityDeepResearch,
  perplexityStandardResearch,
  perplexityRetryResearch,
  validateRetry,
  mergeResearch,
  buildClaudePrompt,
  buildBatchUserPrompt,
  stitchBatchedArticle,
  generateWithClaude,
  checkClaudeResponse,
  generateWithGrok,
  validateStructure,
  extractClaims,
  processVerification,
  claudeAuditAndModerate,
  claudeModerate,
  moderationScan,
  extractAuditResults,
  finalAssembly,
  enrichSlides,
  prepareInputSubjective,
  mergeSubjectiveResearch,
  buildSubjectivePrompt,
  buildSubjectiveBatchUserPrompt,
  generateSubjectiveWithClaude,
  checkSubjectiveClaudeResponse,
  generateSubjectiveWithGrok,
  validateSubjective,
  extractSubjectiveAudit,
  finalAssemblySubjective,
} = proxyActivities<typeof activities>({
  startToCloseTimeout: '3 minutes',
  retry: {
    maximumAttempts: 3,
    initialInterval: '2s',
    backoffCoefficient: 2,
    maximumInterval: '30s',
  },
});

const {
  generateWithGrok: grokLong,
  generateWithClaude: generateWithClaudeLong,
  grokFactCheck: grokFactCheckLong,
  grokSubjectiveStyleAudit: grokSubjectiveStyleAuditLong,
} = proxyActivities<typeof activities>({
  startToCloseTimeout: '8 minutes',
  retry: { maximumAttempts: 2, initialInterval: '5s', backoffCoefficient: 2, maximumInterval: '60s' },
});

// ── Query + Signal definitions (module-level, deterministic) ─────────────────

export const getProgressQuery = defineQuery<WorkflowProgress>('getProgress');
export const humanDecisionSignal = defineSignal<[HumanDecision]>('humanDecision');

// ── Shared post-processing helpers (used by both modes) ──────────────────────

// ── Structured-field image search (replaces old proper-noun extraction) ───────
// Ported from Next.js slide-enricher + ai-generate buildImageSearch pipeline.
// Claude Haiku extracts 7 fields per slide → sanitizeSlide cleans them →
// buildImageSearchFromFields assembles the final query string.

const VALID_EMOTIONS = new Set([
  'celebration', 'defeat', 'focus', 'concern', 'confident', 'intense', 'reflection',
]);

const POSITION_BIGRAMS = new Set([
  'running back', 'wide receiver', 'tight end', 'offensive lineman', 'defensive end',
  'head coach', 'assistant coach', 'offensive coordinator', 'defensive coordinator',
  'point guard', 'shooting guard', 'small forward', 'power forward', 'center fielder',
  'starting pitcher', 'relief pitcher', 'designated hitter', 'free safety', 'strong safety',
  'middle linebacker', 'outside linebacker', 'cornerback', 'quarterback', 'linebacker',
  'general manager', 'team owner', 'team president',
]);

const BANNED_TOKENS = new Set([
  'nfl', 'nba', 'mlb', 'nhl', 'nascar', 'f1', 'formula', 'ufc', 'mma', 'pga',
  'wnba', 'ncaa', 'cfb', 'atp', 'wta', 'fifa',
  'back-to-back', 'record-breaking', 'historic', 'iconic', 'legendary', 'famous',
  'greatest', 'all-time', 'unprecedented', 'remarkable',
]);

const BANNED_TOKENS_EVENT = new Set([
  ...BANNED_TOKENS,
  'championship', 'tournament', 'season', 'final', 'finals', 'round',
]);

function toCleanString(raw: unknown): string {
  if (raw === null || raw === undefined) return '';
  const s = String(raw).trim();
  if (!s || s === 'null' || s === 'undefined' || s === 'N/A' || s === 'n/a') return '';
  return s;
}

function cleanName(raw: string): string {
  let name = toCleanString(raw);
  if (!name) return '';
  // Strip position labels (case-insensitive)
  const lower = name.toLowerCase();
  for (const pos of POSITION_BIGRAMS) {
    if (lower.startsWith(pos + ' ')) {
      name = name.slice(pos.length).trim();
      break;
    }
  }
  // Keep letters (including accented), spaces, apostrophes, hyphens, periods
  name = name.replace(/[^a-zA-ZÀ-ɏ\s'.\-]/g, '').trim();
  // Must have at least first + last name (2 parts)
  const parts = name.split(/\s+/).filter(Boolean);
  if (parts.length < 2) return ''; // surname-only → reject
  return parts.join(' ');
}

function cleanField(raw: string, extraBanned?: Set<string>): string {
  const field = toCleanString(raw);
  if (!field) return '';
  const banned = extraBanned ?? BANNED_TOKENS;
  const words = field.split(/\s+/);
  const cleaned = words.filter(w => !banned.has(w.toLowerCase().replace(/[^a-z0-9-]/g, '')));
  return cleaned.join(' ').trim();
}

function cleanYear(raw: unknown): string {
  const s = toCleanString(raw);
  if (!s) return '';
  const m = s.match(/\b(19|20)\d{2}\b/);
  return m ? m[0] : '';
}

function cleanEmotion(raw: unknown): string {
  const s = toCleanString(raw).toLowerCase().trim();
  return VALID_EMOTIONS.has(s) ? s : '';
}

function cleanOtherSubjects(raw: unknown, excludePrimary: string): string[] {
  if (!Array.isArray(raw)) return [];
  const primaryLower = excludePrimary.toLowerCase();
  return raw
    .map(item => cleanName(String(item ?? '')))
    .filter(name => name && name.toLowerCase() !== primaryLower)
    .slice(0, 3);
}

function expandSurnameFromText(surname: string, title: string, description: string): string {
  if (!surname || surname.includes(' ')) return surname; // already full name or empty
  const text = `${title} ${description}`;
  // Look for "FirstName Surname" pattern in the text
  const escaped = surname.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const regex = new RegExp(`([A-Z][a-z]+)\\s+${escaped}\\b`);
  const m = text.match(regex);
  return m ? `${m[1]} ${surname}` : '';
}

interface EnrichedSlideFields {
  primarySubject: string;
  otherSubjects: string[];
  teamName: string;
  eventName: string;
  location: string;
  year: string;
  emotion: string;
}

function sanitizeSlideFields(
  slide: { title: string; description: string },
  raw: EnrichedSlideFields,
): { imageSearch: string } & EnrichedSlideFields {
  let primarySubject = cleanName(raw.primarySubject);
  const teamName = cleanField(raw.teamName);
  const eventName = cleanField(raw.eventName, BANNED_TOKENS_EVENT);
  const location = cleanField(raw.location);
  const year = cleanYear(raw.year);
  const emotion = cleanEmotion(raw.emotion);

  // Recover surname-only primarySubject by expanding from title+description
  if (!primarySubject && raw.primarySubject) {
    const surname = toCleanString(raw.primarySubject);
    if (surname && !surname.includes(' ')) {
      primarySubject = expandSurnameFromText(surname, slide.title, slide.description);
    }
  }

  const otherSubjects = cleanOtherSubjects(raw.otherSubjects, primarySubject);

  const imageSearch = buildImageSearchFromFields({
    primarySubject, otherSubjects, teamName, eventName, location, year, emotion,
  }) || slide.title || '';

  return { primarySubject, otherSubjects, teamName, eventName, location, year, emotion, imageSearch };
}

function buildImageSearchFromFields(fields: {
  primarySubject: string;
  otherSubjects: string[];
  teamName: string;
  eventName: string;
  location: string;
  year: string;
  emotion: string;
}): string {
  const MAX_TOKENS = 10;
  const seen = new Set<string>();
  const out: string[] = [];

  const addTokens = (raw: unknown) => {
    if (out.length >= MAX_TOKENS) return;
    const s = toCleanString(raw);
    if (!s) return;
    for (const tok of s.split(/\s+/)) {
      if (out.length >= MAX_TOKENS) return;
      const cleaned = tok.replace(/[.,;:!?]+$/g, '').trim();
      if (!cleaned) continue;
      const lo = cleaned.toLowerCase();
      if (seen.has(lo)) continue;
      seen.add(lo);
      out.push(cleaned);
    }
  };

  // Substantive fields first — at least one must produce a token
  addTokens(fields.primarySubject);
  if (Array.isArray(fields.otherSubjects)) {
    for (const o of fields.otherSubjects) addTokens(o);
  }
  addTokens(fields.teamName);
  addTokens(fields.eventName);
  addTokens(fields.location);
  if (out.length === 0) return '';

  // Year + emotion ONLY when a person anchor is present
  const hasPersonAnchor = !!(
    toCleanString(fields.primarySubject) ||
    (Array.isArray(fields.otherSubjects) && fields.otherSubjects.some(o => toCleanString(o)))
  );
  if (hasPersonAnchor) {
    addTokens(fields.year);
    addTokens(fields.emotion);
  }

  return out.join(' ').trim();
}

type ParsedSlide = { slideNum: number; title: string; body: string };

// Parse slides from Claude/Grok output. Resilient to:
//  - Markdown wrapping: **SLIDE N**, ## SLIDE N, ### SLIDE N
//  - Inline title:    "SLIDE 5 The Title On Same Line"
//  - Trailing colon:  "SLIDE 5:"
//  - Missing SOURCES section (just stop at end of text)
//  - Empty body (slide is still emitted so the count is accurate)
function parseSlides(text: string): ParsedSlide[] {
  const result: ParsedSlide[] = [];
  let num: number | null = null;
  let ttl = '';
  let bdy = '';
  const flush = () => {
    if (num !== null) result.push({ slideNum: num, title: ttl.trim(), body: bdy.trim() });
  };
  for (const rawLine of text.split('\n')) {
    // Strip markdown wrappers (**, leading #, trailing colon) so the matcher
    // works on Claude's clean output AND Grok's rewrap variations.
    const t = rawLine.trim().replace(/\*\*/g, '').replace(/^#+\s*/, '').trim();
    const m = t.match(/^SLIDE\s*(\d+)\s*:?\s*(.*)$/i);
    if (m) {
      flush();
      num = parseInt(m[1], 10);
      ttl = m[2].trim();             // captures inline title if present
      bdy = '';
      continue;
    }
    if (num === null) continue;
    if (!t) continue;
    if (t.startsWith('SOURCES')) break;       // end of article body
    if (t.startsWith('META:')) continue;       // META lines never belong to a slide
    if (!ttl) ttl = t;
    else bdy += (bdy ? ' ' : '') + t;
  }
  flush();
  return result;
}

// Split parsed slides into intro + content POSITIONALLY — by where each block
// appears, NOT by its numeric "SLIDE N" label. The first block is always the
// intro; every block after it is content, kept in written order.
//
// Why: Claude sometimes numbers content slides by RANK for countdowns (e.g.
// "SLIDE 9 … SLIDE 1"), which produces a second "SLIDE 1" that collides with the
// intro. The old `find(slideNum===1)` / `filter(slideNum>1)` logic then treated
// the rank-1 item as a duplicate intro and silently dropped it (and the sort
// reversed the countdown order). Positional splitting is immune to whatever
// numbering scheme the model uses.
function splitIntroAndContent(parsed: ParsedSlide[]): { intro: ParsedSlide | undefined; content: ParsedSlide[] } {
  if (parsed.length === 0) return { intro: undefined, content: [] };
  return { intro: parsed[0], content: parsed.slice(1) };
}

interface EnrichmentResult {
  slides: EnrichedSlideFields[];
  status: string;
}

function buildWorkflowResult(
  articleText: string,
  final: {
    title: string; category: string; writerName: string; qualityScore: number;
    summaryComment: string; flagsForReview: string; generatedBy: string;
    moderationFlags?: WorkflowResult['moderationFlags']; moderationVerdict?: WorkflowResult['moderationVerdict'];
  },
  enrichment?: EnrichmentResult | null,
): WorkflowResult {
  const parsedSlides = parseSlides(articleText);
  const { intro: introSlide, content: contentSlides } = splitIntroAndContent(parsedSlides);

  const enrichedContentSlides = contentSlides.map((s, i) => {
    const base = { title: s.title, description: s.body };

    if (enrichment?.status === 'success' && enrichment.slides[i]) {
      const sanitized = sanitizeSlideFields(base, enrichment.slides[i]);
      return { ...base, ...sanitized };
    }

    // Fallback: use slide title as imageSearch (no structured fields)
    return { ...base, imageSearch: s.title || '' };
  });

  const metaMatch = articleText.match(/META:\s*([^\n]+)/i);
  const metaDescription = metaMatch ? metaMatch[1].replace(/\*\*/g, '').trim() : '';

  return {
    title: final.title,
    metaDescription,
    // Editorial rule: the intro slide title mirrors the slideshow title exactly.
    // The model's creative intro headline is discarded; only its body is kept.
    introSlide: introSlide ? { title: final.title, body: introSlide.body } : null,
    description: metaDescription || introSlide?.body || '',
    keywords: final.category,
    slides: enrichedContentSlides,
    author: final.writerName,
    qualityScore: final.qualityScore,
    summaryComment: final.summaryComment,
    flagsForReview: final.flagsForReview,
    generatedBy: final.generatedBy,
    enrichmentStatus: enrichment?.status ?? 'skipped',
    moderationFlags: final.moderationFlags ?? [],
    moderationVerdict: final.moderationVerdict ?? 'PASS',
  };
}

// ── Main workflow ─────────────────────────────────────────────────────────────

export async function msnArticleGeneratorWorkflow(input: FormInput): Promise<WorkflowResult> {

  const mode = input.mode === 'subjective' ? 'subjective' : 'objective';
  const stageDefs = mode === 'subjective' ? STAGE_DEFS_SUBJECTIVE : STAGE_DEFS;

  // ── Stage state ──────────────────────────────────────────────────────────────
  const stages: Stage[] = stageDefs.map(s => ({ ...s, status: 'pending' as const }));
  let currentStageId = stageDefs[0].id;
  let humanReviewRequest: HumanReviewRequest | null = null;
  let humanDecisionReceived: HumanDecision | null = null;
  let workflowResult: WorkflowResult | undefined;

  const idx = (id: string) => stages.findIndex(s => s.id === id);

  const activate = (id: string, detail?: string) => {
    const i = idx(id);
    if (i >= 0) {
      stages[i].status = 'active';
      if (detail) stages[i].detail = detail;
    }
    currentStageId = id;
  };

  const complete = (id: string, detail?: string) => {
    const i = idx(id);
    if (i >= 0) {
      stages[i].status = 'complete';
      if (detail) stages[i].detail = detail;
    }
  };

  const fail = (id: string, detail: string) => {
    const i = idx(id);
    if (i >= 0) {
      stages[i].status = 'error';
      stages[i].detail = detail;
    }
  };

  const warn = (id: string, detail: string) => {
    const i = idx(id);
    if (i >= 0) {
      stages[i].status = 'warning';
      stages[i].detail = detail;
    }
  };

  // Pauses the stage, waits for human signal (30min timeout → auto-continue)
  const askHuman = async (request: HumanReviewRequest): Promise<HumanDecision> => {
    humanReviewRequest = request;
    const i = idx(currentStageId);
    if (i >= 0) stages[i].status = 'awaiting_human';
    humanDecisionReceived = null;

    const gotDecision = await condition(() => humanDecisionReceived !== null, '30 minutes');
    humanReviewRequest = null;

    if (!gotDecision || humanDecisionReceived === null) {
      return { requestId: request.id, choice: 'continue' };
    }
    return humanDecisionReceived;
  };

  // ── Register handlers BEFORE any await ───────────────────────────────────────
  setHandler(getProgressQuery, (): WorkflowProgress => ({
    stages,
    currentStageId,
    humanReviewRequest,
    isComplete: stages[idx('complete')]?.status === 'complete',
    result: workflowResult,
  }));

  setHandler(humanDecisionSignal, (decision: HumanDecision) => {
    humanDecisionReceived = decision;
  });

  // ════════════════════════════════════════════════════════════════════════════
  // SUBJECTIVE MODE — voice-driven, no fact verification, no human pauses.
  // Mirrors n8n "VX Subjective" workflow.
  // ════════════════════════════════════════════════════════════════════════════

  if (mode === 'subjective') {
    // Stage 1: parse
    activate('parsing');
    const prepared = await prepareInputSubjective(input);
    complete('parsing', `${prepared.slideCount} slides · ${prepared.category} · ${prepared.articleType}`);

    // Stage 2: scrape user URLs (up to 5: primary + 4 extras)
    activate('scraping_source');
    let primaryMarkdown = '';
    if (prepared.shouldScrapePrimary) {
      try { primaryMarkdown = await firecrawlScrape(prepared.userPrimaryUrl); }
      catch { warn('scraping_source', 'Primary scrape failed — continuing without it'); }
    }
    const additionalMarkdowns: string[] = [];
    for (const url of prepared.userSecondaryUrls) {
      try {
        const md = await firecrawlScrape(url);
        additionalMarkdowns.push(md ?? '');
      } catch {
        additionalMarkdowns.push('');
      }
    }
    const totalScraped = primaryMarkdown.length + additionalMarkdowns.reduce((s, m) => s + m.length, 0);
    complete('scraping_source', totalScraped > 0 ? `${totalScraped} chars across ${1 + additionalMarkdowns.filter(Boolean).length} source(s)` : 'no source scraped');

    // Stage 3: source alignment analysis (reuse objective activities)
    activate('analyzing');
    let sourcedData;
    if (prepared.hasValidUserSource) {
      sourcedData = await analyzeSourceAlignment(prepared as any, primaryMarkdown, additionalMarkdowns);
    } else {
      sourcedData = await buildResearchStrategy(prepared as any);
    }
    complete('analyzing', `alignment ${sourcedData.sourceAnalysis.alignmentScore}/100 · ${sourcedData.sourceAnalysis.status}`);

    // Stage 4: fact atomization (reuse objective activity)
    activate('atomizing');
    const atomizedData = await atomizeFacts(sourcedData as any);
    complete('atomizing', `${atomizedData.atomizationStats.itemsProcessed} items · ${atomizedData.atomizationStats.totalFacts} facts`);

    // Stage 5: Perplexity research + retry + merge
    activate('researching');
    // Cost optimization: the deep path now uses `sonar` (downgraded from sonar-pro) and
    // only fires for very large slideshows, missing user source, or correlation titles.
    // Raised the slide threshold 15 -> 25 and dropped the mustInclude>=8 clause.
    const needsDeep =
      atomizedData.slideCount > 25 ||
      !atomizedData.hasValidUserSource ||
      (atomizedData as any).titleAnalysis?.requiresCorrelation;

    const emptyPerplexity = { choices: [{ message: { content: '' } }], citations: [] };
    let perplexityRaw;
    try {
      perplexityRaw = needsDeep
        ? await perplexityDeepResearch(atomizedData as any)
        : await perplexityStandardResearch(atomizedData as any);
    } catch {
      warn('researching', 'Perplexity research failed — continuing with source material only');
      perplexityRaw = emptyPerplexity;
    }

    const researchedData = await validateRetry(atomizedData as any, perplexityRaw);

    let merged;
    if (researchedData.needsRetry) {
      let retryRaw;
      try {
        retryRaw = await perplexityRetryResearch(atomizedData as any);
      } catch {
        warn('researching', 'Perplexity retry failed — continuing without retry data');
        retryRaw = emptyPerplexity;
      }
      merged = await mergeSubjectiveResearch(researchedData as any, retryRaw, primaryMarkdown);
    } else {
      merged = await mergeSubjectiveResearch(researchedData as any, undefined, primaryMarkdown);
    }
    complete('researching', `${merged.researchWordCount} words · ${merged.citations.length} citations`);

    // Stage 6: citation scraping
    activate('scraping_citations');
    const skipDomains = [
      // X/Twitter allowed — Firecrawl v2 scrapes it cleanly (omitted from skip list).
      'reddit.com', 'facebook.com', 'instagram.com',
      'tiktok.com', 'pinterest.com', 'quora.com', 'youtube.com', 'youtu.be',
      'nytimes.com', 'wsj.com', 'bloomberg.com', 'theathletic.com', 'linkedin.com',
    ];
    function pickCitationSubj(citations: string[], primary: string, skip: string[]): string {
      const excluded = [...skipDomains, ...skip];
      return citations.find(
        u => u?.startsWith('http') && u !== primary && !excluded.some(d => u.toLowerCase().includes(d))
      ) ?? 'https://www.espn.com';
    }
    const cite1Url = pickCitationSubj(merged.citations, merged.primarySourceUrl, []);
    const cite2Url = pickCitationSubj(merged.citations, merged.primarySourceUrl, [cite1Url]);
    let cite1Markdown = '';
    let cite2Markdown = '';
    try { cite1Markdown = await firecrawlScrape(cite1Url); } catch { /* skip */ }
    try { cite2Markdown = await firecrawlScrape(cite2Url); } catch { /* skip */ }
    complete('scraping_citations', `${cite1Url.split('/')[2]} + ${cite2Url.split('/')[2]}`);

    // Stage 7: build prompt
    activate('building_prompt');
    const prompted = await buildSubjectivePrompt(merged as any);
    complete('building_prompt', `source quality: ${merged.sourceQuality}`);

    // Stage 8: generate (Claude → Grok fallback)
    let generated;

    if (prompted.slideCount > 25) {
      // ── Batched generation (large subjective articles) ─────────────────────
      // Same anti-decay strategy as the objective pipeline: chunks of ≤16
      // content slides, each batch getting full model attention, with per-batch
      // Grok fallback and prior-title continuity.
      const totalContent = prompted.slideCount;
      const maxPerBatch = 16;
      const totalBatches = Math.ceil(totalContent / maxPerBatch);
      const perBatch = Math.ceil(totalContent / totalBatches);

      const batchTexts: string[] = [];
      const batchSources: string[] = [];
      let priorText = '';
      let cursor = 1;
      for (let b = 0; b < totalBatches; b++) {
        const contentCount = Math.min(perBatch, totalContent - (cursor - 1));
        const spec = {
          batchIndex: b,
          totalBatches,
          contentStart: cursor,
          contentCount,
          totalContent,
          isFirst: b === 0,
        };
        activate('generating', `Batch ${b + 1}/${totalBatches} (slides ${cursor}-${cursor + contentCount - 1})…`);
        // factBlock (the fact DB) is identical across every batch of this article;
        // assignment is the per-batch part. Caching the fact block lets batches 2..N
        // read it instead of re-paying full input price.
        const { factBlock, assignment } = await buildSubjectiveBatchUserPrompt(prompted, spec, priorText);
        const grokBatchPrompt = factBlock + assignment;   // Grok auto-caches; send the full prompt

        let batchText = '';
        try {
          const raw = await generateSubjectiveWithClaude(prompted.claudeSystemPrompt, assignment, factBlock);
          const checked = await checkSubjectiveClaudeResponse(raw, prompted);
          if (checked.claudeFailed) {
            batchText = await generateSubjectiveWithGrok(prompted.claudeSystemPrompt, grokBatchPrompt);
            batchSources.push('Grok');
          } else {
            batchText = checked.articleText;
            batchSources.push('Claude');
          }
        } catch {
          try {
            batchText = await generateSubjectiveWithGrok(prompted.claudeSystemPrompt, grokBatchPrompt);
            batchSources.push('Grok');
          } catch {
            throw new Error(`Both Claude and Grok failed on subjective batch ${b + 1}/${totalBatches}`);
          }
        }

        batchTexts.push(batchText);
        // Accumulate ALL prior batches (not just the last) so batch 3+ sees every
        // item already written and can't repeat an early-batch entry.
        priorText = priorText ? `${priorText}\n\n${batchText}` : batchText;
        cursor += contentCount;
      }

      const stitched = await stitchBatchedArticle(batchTexts, { isMultiSlideFormat: prompted.formatConfig.isMultiSlideFormat });
      const allGrok = batchSources.every(s => s === 'Grok');
      const anyGrok = batchSources.some(s => s === 'Grok');
      generated = {
        ...prompted,
        articleText: stitched.articleText,
        originalArticleText: stitched.articleText,
        generatedBy: allGrok ? 'Grok (Fallback, batched)' : anyGrok ? 'Claude + Grok (batched)' : 'Claude (batched)',
        claudeFailed: false,
        failureReason: null,
      };
      complete('generating', `Generated by ${generated.generatedBy} · ${totalBatches} batches`);
    } else {
      // ── Single-shot generation (≤25 slides) ────────────────────────────────
      activate('generating', 'Calling Claude…');
      try {
        const claudeRaw = await generateSubjectiveWithClaude(prompted.claudeSystemPrompt, prompted.claudeUserPrompt);
        const checked = await checkSubjectiveClaudeResponse(claudeRaw, prompted);

        if (checked.claudeFailed) {
          activate('generating', 'Claude failed — falling back to Grok…');
          try {
            const grokText = await generateSubjectiveWithGrok(prompted.claudeSystemPrompt, prompted.claudeUserPrompt);
            generated = {
              ...prompted,
              articleText: grokText,
              originalArticleText: grokText,
              generatedBy: 'Grok (Fallback)',
              claudeFailed: true,
              failureReason: checked.failureReason,
            };
          } catch {
            throw new Error('Both Claude and Grok generation failed — cannot produce subjective article');
          }
        } else {
          generated = checked;
        }
      } catch (err) {
        activate('generating', 'Claude API error — falling back to Grok…');
        try {
          const grokText = await generateSubjectiveWithGrok(prompted.claudeSystemPrompt, prompted.claudeUserPrompt);
          generated = {
            ...prompted,
            articleText: grokText,
            originalArticleText: grokText,
            generatedBy: 'Grok (Fallback)',
            claudeFailed: true,
            failureReason: String(err),
          };
        } catch {
          throw new Error('Both Claude and Grok generation failed — cannot produce subjective article');
        }
      }
      complete('generating', `Generated by ${generated.generatedBy}`);
    }

    // Stage 9: validate structure
    activate('validating');
    const validated = await validateSubjective(generated as any);
    complete('validating', `${validated.slideResults.length} slides · ${validated.errors.length} errors · ${validated.warnings.length} warnings`);

    // Stage 10: Grok style audit
    activate('auditing', 'Running Grok style audit…');
    let audited;
    try {
      const grokAuditText = await grokSubjectiveStyleAuditLong(validated);
      audited = await extractSubjectiveAudit(validated, grokAuditText);
    } catch {
      warn('auditing', 'Grok audit failed — using original article');
      audited = await extractSubjectiveAudit(validated, '');
    }
    complete('auditing', audited.wasAudited ? 'Audited' : 'Skipped');

    // Stage 10b: MSN moderation — parity with the objective pipeline. Haiku
    // contextual pass (claudeModerate) + deterministic scan (moderationScan),
    // merged into a verdict. Flag-only, non-blocking — this is what was missing
    // for subjective articles (they previously shipped a hardcoded PASS).
    try {
      const claudeFlags = await claudeModerate({ title: audited.title, articleText: audited.articleText, category: audited.category });
      const mod = await moderationScan({ title: audited.title, articleText: audited.articleText, moderationFlags: claudeFlags });
      audited = { ...audited, moderationFlags: mod.moderationFlags, moderationVerdict: mod.moderationVerdict };
    } catch {
      warn('auditing', 'Moderation pass failed — continuing without moderation flags');
    }

    // Stage 11: final assembly
    activate('creating_docs', 'Assembling output…');
    const final = await finalAssemblySubjective(audited);
    complete('creating_docs', `score ${final.qualityScore}/100 · moderation ${audited.moderationVerdict ?? 'PASS'} (${(audited.moderationFlags ?? []).length} flags)`);

    // Stage 12: Slide enrichment (Claude Haiku structured-field extraction)
    activate('enriching', 'Extracting image search fields…');
    const parsedForEnrich = splitIntroAndContent(parseSlides(audited.articleText)).content
      .map(s => ({ title: s.title, description: s.body }));

    let enrichment: EnrichmentResult | null = null;
    try {
      enrichment = await enrichSlides(parsedForEnrich, final.title, final.category);
      complete('enriching', `${enrichment.status} · ${enrichment.slides.length} slides`);
    } catch {
      warn('enriching', 'Enrichment failed — using fallback');
    }

    workflowResult = buildWorkflowResult(audited.articleText, final, enrichment);
    complete('complete');
    return workflowResult;
  }

  // ════════════════════════════════════════════════════════════════════════════
  // OBJECTIVE MODE (default) — full fact-verified pipeline with human pauses.
  // ════════════════════════════════════════════════════════════════════════════

  // ── Stage 1: Parse & analyze ─────────────────────────────────────────────────
  activate('parsing');
  const preparedData = await prepareInputAndAnalyze(input);
  complete('parsing', `${preparedData.slideCount} slides · ${preparedData.category}`);

  // ── Stage 2: Source scraping ─────────────────────────────────────────────────
  activate('scraping_source');
  let sourcedData;

  if (preparedData.hasValidUserSource) {
    let primaryMarkdown = '';
    try {
      primaryMarkdown = await firecrawlScrape(preparedData.userPrimaryUrl);
    } catch {
      warn('scraping_source', 'Primary source scrape failed — continuing without it');
    }

    const additionalMarkdowns: string[] = [];
    for (const url of preparedData.userSecondaryUrls) {
      try {
        const md = await firecrawlScrape(url);
        if (md) additionalMarkdowns.push(md);
        else additionalMarkdowns.push('');
      } catch {
        additionalMarkdowns.push('');
      }
    }
    sourcedData = await analyzeSourceAlignment(preparedData, primaryMarkdown, additionalMarkdowns);
  } else {
    sourcedData = await buildResearchStrategy(preparedData);
  }

  complete('scraping_source', `score ${sourcedData.sourceAnalysis.alignmentScore}/100`);

  // ── Human review: poor source alignment ─────────────────────────────────────
  activate('analyzing');
  if (sourcedData.sourceAnalysis.alignmentScore < 40) {
    warn('analyzing', `Low alignment: ${sourcedData.sourceAnalysis.alignmentScore}/100`);
    const decision = await askHuman({
      id: `poor-source-${Date.now()}`,
      type: 'poor_source',
      message: 'Source alignment is low — the scraped content may not match the article title.',
      details: [
        `Alignment score: ${sourcedData.sourceAnalysis.alignmentScore}/100`,
        `Status: ${sourcedData.sourceAnalysis.status}`,
        `Recommendation: ${sourcedData.sourceAnalysis.recommendation}`,
      ],
      options: [
        { label: 'Continue anyway', value: 'continue' },
        { label: 'Swap to research-only mode (no user source)', value: 'research_only' },
        {
          label: 'Provide a better source URL',
          value: 'new_source',
          requiresInput: true,
          inputPlaceholder: 'Paste a better source URL…',
        },
      ],
    });

    if (decision.choice === 'research_only') {
      sourcedData = await buildResearchStrategy(preparedData);
    }
    // 'new_source' with additionalInput would need a re-scrape — handled in the
    // next iteration; for now fall through to continue with original data
  }
  complete('analyzing', `${sourcedData.sourceAnalysis.status}`);

  // ── Stage 3: Fact atomization ────────────────────────────────────────────────
  activate('atomizing');
  const atomizedData = await atomizeFacts(sourcedData);
  complete('atomizing', `${atomizedData.atomizationStats.itemsProcessed} items · ${atomizedData.atomizationStats.totalFacts} facts`);

  // ── Stage 4: Perplexity research ─────────────────────────────────────────────
  activate('researching');
  // Cost optimization: the deep path now uses `sonar` (downgraded from sonar-pro) and
  // only fires for very large slideshows, missing user source, or correlation titles.
  // Raised the slide threshold 15 -> 25 and dropped the mustInclude>=8 clause.
  const needsDeep =
    atomizedData.slideCount > 25 ||
    !atomizedData.hasValidUserSource ||
    atomizedData.titleAnalysis.requiresCorrelation;

  const emptyPerplexity = { choices: [{ message: { content: '' } }], citations: [] };
  let perplexityRaw;
  try {
    perplexityRaw = needsDeep
      ? await perplexityDeepResearch(atomizedData)
      : await perplexityStandardResearch(atomizedData);
  } catch {
    warn('researching', 'Perplexity research failed — continuing with source material only');
    perplexityRaw = emptyPerplexity;
  }

  // ── Stage 5: Retry validation ────────────────────────────────────────────────
  const researchedData = await validateRetry(atomizedData, perplexityRaw);

  let mergedData;
  if (researchedData.needsRetry) {
    let retryRaw;
    try {
      retryRaw = await perplexityRetryResearch(atomizedData);
    } catch {
      warn('researching', 'Perplexity retry failed — continuing without retry data');
      retryRaw = emptyPerplexity;
    }
    mergedData = await mergeResearch(researchedData, retryRaw);
  } else {
    mergedData = await mergeResearch(researchedData);
  }

  complete('researching', `${mergedData.researchWordCount} words · ${mergedData.citations.length} citations`);

  // ── Human review: thin research after retry ──────────────────────────────────
  if (!mergedData.researchOk) {
    warn('researching', 'Research thin after retry');
    const decision = await askHuman({
      id: `thin-research-${Date.now()}`,
      type: 'thin_research',
      message: 'Perplexity research is thin even after a retry. The article may lack depth.',
      details: [
        `Research word count: ${mergedData.researchWordCount}`,
        `Citations found: ${mergedData.citations.length}`,
        researchedData.retryReason ?? 'No retry reason captured',
      ],
      options: [
        { label: 'Proceed with thin research', value: 'continue' },
        { label: 'Abort and try a different topic', value: 'abort' },
        {
          label: 'Add context to supplement research',
          value: 'add_context',
          requiresInput: true,
          inputPlaceholder: 'Add extra context or facts…',
        },
      ],
    });

    if (decision.choice === 'abort') {
      throw new Error('Workflow aborted by human: thin research');
    }
  }

  // ── Stage 6: Citation scraping ───────────────────────────────────────────────
  activate('scraping_citations');

  const skipDomains = [
    // X/Twitter allowed — Firecrawl v2 scrapes it cleanly (omitted from skip list).
    'reddit.com', 'facebook.com', 'instagram.com',
    'tiktok.com', 'pinterest.com', 'quora.com', 'youtube.com', 'youtu.be',
    'nytimes.com', 'wsj.com', 'bloomberg.com', 'theathletic.com', 'linkedin.com',
  ];

  function pickCitation(citations: string[], primary: string, skip: string[]): string {
    const excluded = [...skipDomains, ...skip];
    return citations.find(
      u => u?.startsWith('http') && u !== primary && !excluded.some(d => u.toLowerCase().includes(d))
    ) ?? 'https://www.espn.com';
  }

  const cite1Url = pickCitation(mergedData.citations, mergedData.primarySourceUrl, []);
  const cite2Url = pickCitation(mergedData.citations, mergedData.primarySourceUrl, [cite1Url]);

  let cite1Markdown = '';
  let cite2Markdown = '';
  try { cite1Markdown = await firecrawlScrape(cite1Url); } catch { /* skip on failure */ }
  try { cite2Markdown = await firecrawlScrape(cite2Url); } catch { /* skip on failure */ }
  complete('scraping_citations', `${cite1Url.split('/')[2]} + ${cite2Url.split('/')[2]}`);

  // ── Stage 7: Build Claude prompt ─────────────────────────────────────────────
  activate('building_prompt');
  const promptData = await buildClaudePrompt(mergedData, cite1Markdown, cite2Markdown);
  complete('building_prompt');

  // ── Stage 8 + 9: Article generation → Structural validation ─────────────────
  // Wrapped in a loop so a human "regenerate" signal can retry once.
  let validatedData!: Awaited<ReturnType<typeof validateStructure>>;
  let generationAttempt = 0;
  const MAX_GENERATION_ATTEMPTS = 2;

  while (generationAttempt < MAX_GENERATION_ATTEMPTS) {
    generationAttempt++;

    let generatedData;

    if (promptData.slideCount > 25) {
      // ── Batched generation (large articles) ──────────────────────────────────
      // Generate in chunks of ≤16 content slides so each call gets full model
      // attention and the back half doesn't degrade. Each batch is small enough
      // to use the fast non-streaming path. Per-batch Grok fallback keeps quality
      // parity if a Claude batch fails.
      const totalContent = promptData.slideCount;
      const maxPerBatch = 16;
      const totalBatches = Math.ceil(totalContent / maxPerBatch);
      const perBatch = Math.ceil(totalContent / totalBatches);

      activate('generating', generationAttempt > 1 ? `Regenerating in ${totalBatches} batches…` : `Generating in ${totalBatches} batches…`);

      const batchTexts: string[] = [];
      const batchSources: string[] = [];
      let priorText = '';
      let cursor = 1;
      for (let b = 0; b < totalBatches; b++) {
        const contentCount = Math.min(perBatch, totalContent - (cursor - 1));
        const spec = {
          batchIndex: b,
          totalBatches,
          contentStart: cursor,
          contentCount,
          totalContent,
          isFirst: b === 0,
        };
        activate('generating', `Batch ${b + 1}/${totalBatches} (slides ${cursor}-${cursor + contentCount - 1})…`);
        // factBlock (the fact DB) is identical across every batch of this article;
        // assignment is the per-batch part. Caching the fact block lets batches 2..N
        // read it instead of re-paying full input price.
        const { factBlock, assignment } = await buildBatchUserPrompt(promptData, spec, priorText);
        const grokBatchPrompt = factBlock + assignment;   // Grok auto-caches; send the full prompt

        let batchText = '';
        try {
          // contentCount ≤ 16, so this stays on the fast non-streaming path.
          const raw = await generateWithClaude(promptData.claudeSystemPrompt, assignment, contentCount, factBlock);
          const checked = await checkClaudeResponse(raw, promptData);
          if (checked.claudeFailed) {
            batchText = await grokLong(promptData.claudeSystemPrompt, grokBatchPrompt);
            batchSources.push('Grok');
          } else {
            batchText = checked.articleText;
            batchSources.push('Claude');
          }
        } catch {
          try {
            batchText = await grokLong(promptData.claudeSystemPrompt, grokBatchPrompt);
            batchSources.push('Grok');
          } catch {
            throw new Error(`Both Claude and Grok failed on batch ${b + 1}/${totalBatches} — cannot produce article`);
          }
        }

        batchTexts.push(batchText);
        // Accumulate ALL prior batches (not just the last) so batch 3+ sees every
        // item already written and can't repeat an early-batch entry.
        priorText = priorText ? `${priorText}\n\n${batchText}` : batchText;
        cursor += contentCount;
      }

      const stitched = await stitchBatchedArticle(batchTexts, { isMultiSlideFormat: promptData.formatConfig.isMultiSlideFormat });
      const allGrok = batchSources.every(s => s === 'Grok');
      const anyGrok = batchSources.some(s => s === 'Grok');
      generatedData = {
        ...promptData,
        articleText: stitched.articleText,
        originalArticleText: stitched.articleText,
        generatedBy: allGrok ? 'Grok (Fallback, batched)' : anyGrok ? 'Claude + Grok (batched)' : 'Claude (batched)',
        claudeFailed: false,
        failureReason: null,
      };
      complete('generating', `Generated by ${generatedData.generatedBy} · ${totalBatches} batches`);
    } else {
      // ── Single-shot generation (≤25 slides) ──────────────────────────────────
      // Stream very large single-shot articles (>30) via the long-timeout proxy so
      // a slow generation can't trip the socket timeout; otherwise use the fast
      // non-streaming path. (With batching at >25, the streaming branch is a
      // defensive fallback that won't normally be reached.)
      const useClaudeStreaming = promptData.slideCount > 30;
      const claudeGen = useClaudeStreaming ? generateWithClaudeLong : generateWithClaude;

      activate('generating', generationAttempt > 1 ? 'Regenerating article…' : (useClaudeStreaming ? 'Calling Claude (streaming)…' : 'Calling Claude…'));
      try {
        const claudeRaw = await claudeGen(promptData.claudeSystemPrompt, promptData.claudeUserPrompt, promptData.slideCount);
        const claudeChecked = await checkClaudeResponse(claudeRaw, promptData);

        if (claudeChecked.claudeFailed) {
          activate('generating', 'Claude failed — falling back to Grok…');
          try {
            const grokText = await grokLong(promptData.claudeSystemPrompt, promptData.claudeUserPrompt);
            generatedData = {
              ...promptData,
              articleText: grokText,
              originalArticleText: grokText,
              generatedBy: 'Grok (Fallback)',
              claudeFailed: true,
              failureReason: claudeChecked.failureReason,
            };
          } catch {
            throw new Error('Both Claude and Grok generation failed — cannot produce article');
          }
        } else {
          generatedData = claudeChecked;
        }
      } catch (err) {
        activate('generating', 'Claude API error — falling back to Grok…');
        try {
          const grokText = await grokLong(promptData.claudeSystemPrompt, promptData.claudeUserPrompt);
          generatedData = {
            ...promptData,
            articleText: grokText,
            originalArticleText: grokText,
            generatedBy: 'Grok (Fallback)',
            claudeFailed: true,
            failureReason: String(err),
          };
        } catch {
          throw new Error('Both Claude and Grok generation failed — cannot produce article');
        }
      }
      complete('generating', `Generated by ${generatedData.generatedBy}`);
    }

    // ── Stage 9: Structural validation ─────────────────────────────────────────
    activate('validating');
    validatedData = await validateStructure(generatedData);
    const hardErrors = validatedData.structuralValidation.errors.filter(
      e => !validatedData.structuralValidation.autoFixes.includes(e)
    );
    complete('validating', `${validatedData.structuralValidation.slideCount} slides · ${hardErrors.length} errors`);

    // ── Human review: structural errors ────────────────────────────────────────
    if (hardErrors.length > 0) {
      warn('validating', `${hardErrors.length} errors after auto-fix`);
      const decision = await askHuman({
        id: `validation-errors-${Date.now()}`,
        type: 'validation_errors',
        message: 'The generated article has structural errors that could not be auto-fixed.',
        details: hardErrors.slice(0, 5),
        options: [
          { label: 'Continue with errors (will be flagged)', value: 'continue' },
          ...(generationAttempt < MAX_GENERATION_ATTEMPTS
            ? [{ label: 'Regenerate article', value: 'regenerate' }]
            : []),
          { label: 'Abort', value: 'abort' },
        ],
      });

      if (decision.choice === 'abort') {
        throw new Error('Workflow aborted by human: structural validation errors');
      }
      if (decision.choice === 'regenerate' && generationAttempt < MAX_GENERATION_ATTEMPTS) {
        continue; // loop back to Stage 8
      }
    }
    break; // no errors, or human chose "continue" — proceed
  }

  // ── Stage 10: Claim extraction + Grok fact-check ─────────────────────────────
  activate('verifying');
  const claimedData = await extractClaims(validatedData);

  let verifiedData;
  try {
    const verifyRaw = await grokFactCheckLong(claimedData);
    verifiedData = await processVerification(claimedData, verifyRaw);
  } catch {
    warn('verifying', 'Grok fact-check failed — skipping verification');
    verifiedData = {
      ...claimedData,
      perplexityVerification: {
        results: [],
        citations: [],
        stats: { verified: 0, incorrect: 0, unverifiable: 0, total: 0 },
        score: 0,
      },
    };
  }

  const factErrors = verifiedData.perplexityVerification.stats.incorrect;
  complete('verifying', `${verifiedData.perplexityVerification.stats.verified} verified · ${factErrors} incorrect`);

  // ── Human review: multiple fact errors ───────────────────────────────────────
  if (factErrors > 2) {
    warn('verifying', `${factErrors} facts flagged as incorrect`);
    const decision = await askHuman({
      id: `fact-errors-${Date.now()}`,
      type: 'fact_errors',
      message: `Perplexity found ${factErrors} potentially incorrect facts.`,
      details: verifiedData.perplexityVerification.results
        .filter(r => r.status === 'incorrect')
        .slice(0, 5)
        .map(r => `Claim: "${r.claim}" → ${r.finding}`),
      options: [
        { label: 'Proceed to Grok audit (it will correct these)', value: 'continue' },
        { label: 'Abort and restart', value: 'abort' },
      ],
    });
    if (decision.choice === 'abort') {
      throw new Error('Workflow aborted by human: too many fact errors');
    }
  }

  // ── Stage 11: Claude audit + MSN moderation ────────────────────────────────────
  // Audit/style/moderation now run on Claude Haiku (cheaper, faster, cacheable
  // ruleset). The fact-check above stays on Grok. A deterministic code node
  // (moderationScan) then adds the literal banned-word / typography flags.
  activate('auditing', 'Running Claude audit + moderation…');
  let auditedData;
  try {
    const auditText = await claudeAuditAndModerate(verifiedData);
    auditedData = await extractAuditResults(verifiedData, auditText);
  } catch {
    warn('auditing', 'Claude audit failed — skipping rules audit');
    auditedData = {
      ...verifiedData,
      grokAudit: {
        status: 'skipped',
        rawResponse: '',
        summary: 'Audit skipped due to API failure',
        stats: { rulesPassed: 'N/A', violations: 0, corrections: '0', flags: 'Audit skipped' },
      },
      grokSources: [],
      combinedSourceList: [],
      combinedSourceListText: '',
      rewriteApplied: false,
      moderationFlags: [],
      moderationVerdict: 'PASS' as const,
    };
  }

  // Deterministic moderation scan (banned words, typography, title patterns, length).
  // Runs on the post-audit article so flags reflect what ships; merges with the
  // Claude moderation flags and recomputes the verdict. Never blocks the flow.
  try {
    const mod = await moderationScan({ title: auditedData.title, articleText: auditedData.articleText, moderationFlags: auditedData.moderationFlags });
    auditedData = { ...auditedData, moderationFlags: mod.moderationFlags, moderationVerdict: mod.moderationVerdict };
  } catch {
    warn('auditing', 'Moderation scan failed — keeping Claude flags only');
  }
  complete('auditing', `Rules: ${auditedData.grokAudit.stats.rulesPassed} · Moderation: ${auditedData.moderationVerdict} (${auditedData.moderationFlags.length} flags)`);

  // ── Stage 12: Final assembly ──────────────────────────────────────────────────
  activate('creating_docs', 'Assembling output…');
  const final = await finalAssembly(auditedData);

  // ── Human review: low quality score ──────────────────────────────────────────
  if (final.qualityScore < 55) {
    warn('creating_docs', `Low quality score: ${final.qualityScore}/100`);
    const decision = await askHuman({
      id: `low-quality-${Date.now()}`,
      type: 'low_quality',
      message: `Quality score is ${final.qualityScore}/100 — below the 55-point threshold.`,
      details: [
        `Research score: ${final.researchScore}`,
        `Verification score: ${final.verificationScore}`,
        `Structural score: ${final.structuralScore}`,
        `Originality score: ${final.originalityScore}`,
        final.flagsForReview || 'No specific flags',
      ],
      options: [
        { label: 'Use it anyway', value: 'continue' },
        { label: 'Abort', value: 'abort' },
      ],
    });
    if (decision.choice === 'abort') {
      throw new Error(`Workflow aborted by human: quality score ${final.qualityScore}/100`);
    }
  }

  complete('creating_docs', `score ${final.qualityScore}/100`);

  // ── Stage 13: Slide enrichment (Claude Haiku structured-field extraction) ───
  activate('enriching', 'Extracting image search fields…');
  const parsedForEnrich = splitIntroAndContent(parseSlides(auditedData.articleText)).content
    .map(s => ({ title: s.title, description: s.body }));

  let enrichment: EnrichmentResult | null = null;
  try {
    enrichment = await enrichSlides(parsedForEnrich, final.title, final.category);
    complete('enriching', `${enrichment.status} · ${enrichment.slides.length} slides`);
  } catch {
    warn('enriching', 'Enrichment failed — using fallback');
  }

  workflowResult = buildWorkflowResult(auditedData.articleText, final, enrichment);
  complete('complete');
  return workflowResult;
}
