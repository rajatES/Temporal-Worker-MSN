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
  generateWithClaude,
  checkClaudeResponse,
  generateWithGrok,
  validateStructure,
  extractClaims,
  processVerification,
  grokAuditAndVerify,
  extractAuditResults,
  finalAssembly,
  enrichSlides,
  prepareInputSubjective,
  perplexitySubjectiveContext,
  mergeSubjectiveContext,
  buildSubjectivePrompt,
  generateSubjectiveWithClaude,
  checkSubjectiveClaudeResponse,
  generateSubjectiveWithGrok,
  validateSubjective,
  grokSubjectiveStyleAudit,
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
  grokAuditAndVerify: grokAuditLong,
  generateWithGrok: grokLong,
  grokFactCheck: grokFactCheckLong,
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

interface EnrichmentResult {
  slides: EnrichedSlideFields[];
  status: string;
}

function buildWorkflowResult(
  articleText: string,
  final: { title: string; category: string; writerName: string; qualityScore: number; summaryComment: string; flagsForReview: string; generatedBy: string },
  enrichment?: EnrichmentResult | null,
): WorkflowResult {
  const parsedSlides = parseSlides(articleText);
  const introSlide = parsedSlides.find(s => s.slideNum === 1);
  const contentSlides = parsedSlides
    .filter(s => s.slideNum > 1)
    .sort((a, b) => a.slideNum - b.slideNum);

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
    introSlide: introSlide ? { title: introSlide.title, body: introSlide.body } : null,
    description: metaDescription || introSlide?.body || '',
    keywords: final.category,
    slides: enrichedContentSlides,
    author: final.writerName,
    qualityScore: final.qualityScore,
    summaryComment: final.summaryComment,
    flagsForReview: final.flagsForReview,
    generatedBy: final.generatedBy,
    enrichmentStatus: enrichment?.status ?? 'skipped',
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

    // Stage 3: Perplexity context research
    activate('researching');
    const perplexityResp = await perplexitySubjectiveContext(prepared);
    complete('researching', `${perplexityResp.citations?.length ?? 0} citations`);

    // Stage 4: merge tiers
    activate('building_prompt');
    const merged = await mergeSubjectiveContext(prepared, primaryMarkdown, additionalMarkdowns, perplexityResp);
    const prompted = await buildSubjectivePrompt(merged);
    complete('building_prompt', `source quality: ${merged.sourceQuality}`);

    // Stage 5: generate (Claude → Grok fallback)
    activate('generating', 'Calling Claude…');
    let generated;
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

    // Stage 6: validate structure
    activate('validating');
    const validated = await validateSubjective(generated);
    complete('validating', `${validated.slideResults.length} slides · ${validated.errors.length} errors · ${validated.warnings.length} warnings`);

    // Stage 7: Grok style audit
    activate('auditing', 'Running Grok style audit…');
    let audited;
    try {
      const grokAuditText = await grokSubjectiveStyleAudit(validated);
      audited = await extractSubjectiveAudit(validated, grokAuditText);
    } catch {
      warn('auditing', 'Grok audit failed — using original article');
      audited = await extractSubjectiveAudit(validated, '');
    }
    complete('auditing', audited.wasAudited ? 'Audited' : 'Skipped');

    // Stage 8: final assembly
    activate('creating_docs', 'Assembling output…');
    const final = await finalAssemblySubjective(audited);
    complete('creating_docs', `score ${final.qualityScore}/100`);

    // Stage 9: Slide enrichment (Claude Haiku structured-field extraction)
    activate('enriching', 'Extracting image search fields…');
    const parsedForEnrich = parseSlides(audited.articleText)
      .filter(s => s.slideNum > 1)
      .sort((a, b) => a.slideNum - b.slideNum)
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
  const needsDeep =
    atomizedData.slideCount >= 15 ||
    atomizedData.mustIncludeItems.length >= 8 ||
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
    'twitter.com', 'x.com', 'reddit.com', 'facebook.com', 'instagram.com',
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

    activate('generating', generationAttempt > 1 ? 'Regenerating article…' : 'Calling Claude…');
    let generatedData;
    try {
      const claudeRaw = await generateWithClaude(promptData.claudeSystemPrompt, promptData.claudeUserPrompt);
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

  // ── Stage 11: Grok audit ──────────────────────────────────────────────────────
  activate('auditing', 'Running Grok rules audit…');
  let auditedData;
  try {
    const grokAuditText = await grokAuditLong(verifiedData);
    auditedData = await extractAuditResults(verifiedData, grokAuditText);
  } catch {
    warn('auditing', 'Grok audit failed — skipping rules audit');
    auditedData = {
      ...verifiedData,
      grokAudit: {
        status: 'skipped',
        rawResponse: '',
        summary: 'Grok audit skipped due to API failure',
        stats: { rulesPassed: 'N/A', violations: 0, corrections: '0', flags: 'Audit skipped' },
      },
      grokSources: [],
      combinedSourceList: [],
      combinedSourceListText: '',
      rewriteApplied: false,
    };
  }
  complete('auditing', `Rules: ${auditedData.grokAudit.stats.rulesPassed} · Violations: ${auditedData.grokAudit.stats.violations} · Rewrite: ${auditedData.rewriteApplied}`);

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
  const parsedForEnrich = parseSlides(auditedData.articleText)
    .filter(s => s.slideNum > 1)
    .sort((a, b) => a.slideNum - b.slideNum)
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
