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
  // Subjective pipeline
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

const STOP_WORDS = new Set([
  'a','an','the','and','or','but','in','on','at','to','for','of','with','by','from',
  'is','was','are','were','has','had','have','his','her','their','its','this','that',
  'he','she','they','it','who','which','also','as','be','been','being','into','up',
  'out','over','after','before','when','than','not','no','so','if','would','could',
  'should','two','three','four','five','despite','during','while','through',
  'against','between','among','across','around','behind','below','above','under',
]);
const SPORT_LABELS: Record<string, string> = {
  nfl: 'NFL', nba: 'NBA', mlb: 'MLB', nhl: 'NHL', nascar: 'NASCAR',
  'f1': 'F1', 'formula 1': 'F1', golf: 'Golf', pga: 'Golf',
  ufc: 'UFC', mma: 'MMA', tennis: 'Tennis', atp: 'Tennis', wta: 'Tennis',
  wnba: 'WNBA', ncaa: 'NCAA', cfb: 'CFB', boxing: 'Boxing',
};

function extractProperNouns(text: string): string[] {
  const tokens = text.replace(/\*\*/g, '').split(/\s+/);
  const nouns: string[] = [];
  let i = 0;
  while (i < tokens.length) {
    const raw = tokens[i];
    const clean = raw.replace(/[^a-zA-Z0-9]/g, '');
    if (!clean || clean.length < 2 || STOP_WORDS.has(clean.toLowerCase())) { i++; continue; }
    if (/^[A-Z]/.test(raw)) {
      const parts = [clean];
      let j = i + 1;
      while (j < i + 4 && j < tokens.length) {
        const nr = tokens[j], nc = nr.replace(/[^a-zA-Z0-9]/g, '');
        if (/^[A-Z]/.test(nr) && nc.length > 1 && !STOP_WORDS.has(nc.toLowerCase())) {
          parts.push(nc); j++;
        } else break;
      }
      nouns.push(parts.join(' '));
      i = j;
    } else { i++; }
  }
  const unique: string[] = [];
  const seen = new Set<string>();
  for (const noun of nouns) {
    const key = noun.toLowerCase();
    if (seen.has(key)) continue;
    if (unique.some(u => u.toLowerCase().includes(key))) continue;
    unique.push(noun);
    seen.add(key);
  }
  return unique;
}

function buildImageSearch(title: string, body: string, category: string): string {
  const descNouns = extractProperNouns(body || '');
  let candidates: string[] = [...descNouns];

  if (candidates.length < 2) {
    const titleNouns = extractProperNouns((title || '').replace(/^\d+\.\s*/, ''))
      .map(n => n.split(' ').slice(0, 2).join(' '));
    for (const n of titleNouns) {
      if (!n.includes(' ')) continue;
      if (!candidates.some(d => d.toLowerCase().includes(n.toLowerCase()) || n.toLowerCase().includes(d.toLowerCase()))) {
        candidates.push(n);
      }
    }
    if (candidates.length === 0) {
      const fallback = (title || '').replace(/^\d+\.\s*/, '').replace(/[^a-zA-Z0-9\s]/g, '').trim().split(/\s+/).slice(0, 2).join(' ');
      if (fallback) candidates.push(fallback);
    }
  }

  const seen = new Set<string>();
  const terms: string[] = [];
  for (const t of candidates) {
    const k = t.toLowerCase();
    if (seen.has(k)) continue;
    if (terms.some(u => u.toLowerCase().includes(k) || k.includes(u.toLowerCase()))) continue;
    terms.push(t);
    seen.add(k);
  }
  const catKey = (category || '').toLowerCase().replace('sports - ', '').trim();
  const sport = SPORT_LABELS[catKey] || '';
  if (sport && !terms.some(t => t.toLowerCase().includes(sport.toLowerCase()))) {
    terms.push(sport);
  }
  return terms.slice(0, 5).join(' ').replace(/\s+/g, ' ').trim();
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

function buildWorkflowResult(articleText: string, final: { title: string; category: string; writerName: string; qualityScore: number; summaryComment: string; flagsForReview: string; generatedBy: string }): WorkflowResult {
  const parsedSlides  = parseSlides(articleText);
  const introSlide    = parsedSlides.find(s => s.slideNum === 1);
  const contentSlides = parsedSlides
    .filter(s => s.slideNum > 1)
    .sort((a, b) => a.slideNum - b.slideNum)
    .map(s => ({
      title:       s.title,
      description: s.body,
      imageSearch: buildImageSearch(s.title, s.body, final.category),
    }));

  const metaMatch = articleText.match(/META:\s*([^\n]+)/i);
  const metaDescription = metaMatch ? metaMatch[1].replace(/\*\*/g, '').trim() : '';

  return {
    title:         final.title,
    metaDescription,
    introSlide:    introSlide ? { title: introSlide.title, body: introSlide.body } : null,
    description:   metaDescription || introSlide?.body || '',
    keywords:      final.category,
    slides:        contentSlides,
    author:        final.writerName,
    qualityScore:  final.qualityScore,
    summaryComment: final.summaryComment,
    flagsForReview: final.flagsForReview,
    generatedBy:   final.generatedBy,
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
      const checked   = await checkSubjectiveClaudeResponse(claudeRaw, prompted);

      if (checked.claudeFailed) {
        activate('generating', 'Claude failed — falling back to Grok…');
        try {
          const grokText = await generateSubjectiveWithGrok(prompted.claudeSystemPrompt, prompted.claudeUserPrompt);
          generated = {
            ...prompted,
            articleText:         grokText,
            originalArticleText: grokText,
            generatedBy:         'Grok (Fallback)',
            claudeFailed:        true,
            failureReason:       checked.failureReason,
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
          articleText:         grokText,
          originalArticleText: grokText,
          generatedBy:         'Grok (Fallback)',
          claudeFailed:        true,
          failureReason:       String(err),
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

    workflowResult = buildWorkflowResult(audited.articleText, final);
    complete('creating_docs', `${workflowResult.slides.length} slides · score ${final.qualityScore}/100`);
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

    // Scrape every remaining URL (up to 4) — secondaries use onlyMainContent=false
    // to capture broader context (matches n8n behaviour for non-primary sources).
    const additionalMarkdowns: string[] = [];
    for (const url of preparedData.userSecondaryUrls) {
      try {
        const md = await firecrawlScrape(url, false);
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

  // ── Stage 8: Article generation ──────────────────────────────────────────────
  activate('generating', 'Calling Claude…');
  let generatedData;
  try {
    const claudeRaw     = await generateWithClaude(promptData.claudeSystemPrompt, promptData.claudeUserPrompt);
    const claudeChecked = await checkClaudeResponse(claudeRaw, promptData);

    if (claudeChecked.claudeFailed) {
      activate('generating', 'Claude failed — falling back to Grok…');
      try {
        const grokText = await grokLong(promptData.claudeSystemPrompt, promptData.claudeUserPrompt);
        generatedData = {
          ...promptData,
          articleText:         grokText,
          originalArticleText: grokText,
          generatedBy:         'Grok (Fallback)',
          claudeFailed:        true,
          failureReason:       claudeChecked.failureReason,
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
        articleText:         grokText,
        originalArticleText: grokText,
        generatedBy:         'Grok (Fallback)',
        claudeFailed:        true,
        failureReason:       String(err),
      };
    } catch {
      throw new Error('Both Claude and Grok generation failed — cannot produce article');
    }
  }
  complete('generating', `Generated by ${generatedData.generatedBy}`);

  // ── Stage 9: Structural validation ───────────────────────────────────────────
  activate('validating');
  const validatedData = await validateStructure(generatedData);
  const hardErrors = validatedData.structuralValidation.errors.filter(
    e => !validatedData.structuralValidation.autoFixes.includes(e)
  );
  complete('validating', `${validatedData.structuralValidation.slideCount} slides · ${hardErrors.length} errors`);

  // ── Human review: structural errors ──────────────────────────────────────────
  if (hardErrors.length > 0) {
    warn('validating', `${hardErrors.length} errors after auto-fix`);
    await askHuman({
      id: `validation-errors-${Date.now()}`,
      type: 'validation_errors',
      message: 'The generated article has structural errors that could not be auto-fixed.',
      details: hardErrors.slice(0, 5),
      options: [
        { label: 'Continue with errors (will be flagged)', value: 'continue' },
        { label: 'Regenerate article', value: 'regenerate' },
        { label: 'Abort', value: 'abort' },
      ],
    });
    // Note: 'regenerate' would loop back; for now we continue and flag
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

  workflowResult = buildWorkflowResult(auditedData.articleText, final);
  complete('creating_docs', `${workflowResult.slides.length} slides · score ${final.qualityScore}/100`);
  complete('complete');
  return workflowResult;
}
