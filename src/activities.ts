import axios from 'axios';
import * as dotenv from 'dotenv';
import {
  FormInput, PreparedData, SourcedData, AtomizedData, ResearchedData,
  MergedData, PromptData, GeneratedData, ValidatedData, ClaimedData,
  VerifiedData, AuditedData, FinalOutput, TemporalCtx,
  FormatConfig, TitleAnalysis, AtomizedFact, SourceEntry, FactProvenance,
} from './types';

dotenv.config();

const FIRECRAWL_KEY  = process.env.FIRECRAWL_API_KEY!;
const PERPLEXITY_KEY = process.env.PERPLEXITY_API_KEY!;
const ANTHROPIC_KEY  = process.env.ANTHROPIC_API_KEY!;
const GROK_KEY       = process.env.GROK_API_KEY!;

const RESTRICTED_DOMAINS = [
  'instagram.com', 'facebook.com', 'twitter.com', 'x.com', 'tiktok.com',
  'linkedin.com', 'pinterest.com', 'nytimes.com', 'wsj.com', 'ft.com',
  'bloomberg.com', 'theathletic.com', 'reddit.com', 'quora.com', 'youtube.com',
];

function isRestricted(url: string, extra: string[] = []): boolean {
  const all = [...RESTRICTED_DOMAINS, ...extra];
  return all.some(d => url.toLowerCase().includes(d));
}

// ── 1. prepareInputAndAnalyze (n8n: "Prepare Input & Analyze") ────────────────

export async function prepareInputAndAnalyze(input: FormInput): Promise<PreparedData> {
  const {
    writerName, title: rawTitle, category: rawCategory, slideCount: rawSlides,
    slidesPerEntityRaw, sourcesRaw, mustIncludeRaw, userContext, writingStyle,
  } = input;

  const title    = rawTitle.trim();
  const category = rawCategory.trim();
  const slideCount = rawSlides || 20;

  if (!title)    throw new Error('Slideshow Title is required');
  if (!category) throw new Error('Topic Category is required');

  // Temporal context
  const now          = new Date();
  const currentYear  = now.getFullYear();
  const currentMonth = now.getMonth() + 1;

  type SeasonDef = { currentSeason: string; lastSeason: string; seasonFormat: string };
  const sportSeasons: Record<string, SeasonDef> = {
    NFL:    { currentSeason: currentMonth >= 9 ? `${currentYear}` : `${currentYear - 1}`, lastSeason: currentMonth >= 9 ? `${currentYear - 1}` : `${currentYear - 2}`, seasonFormat: 'single_year' },
    NBA:    { currentSeason: currentMonth >= 10 ? `${currentYear}-${String(currentYear + 1).slice(2)}` : `${currentYear - 1}-${String(currentYear).slice(2)}`, lastSeason: currentMonth >= 10 ? `${currentYear - 1}-${String(currentYear).slice(2)}` : `${currentYear - 2}-${String(currentYear - 1).slice(2)}`, seasonFormat: 'split_year' },
    WNBA:   { currentSeason: currentMonth >= 5 ? `${currentYear}` : `${currentYear - 1}`, lastSeason: currentMonth >= 5 ? `${currentYear - 1}` : `${currentYear - 2}`, seasonFormat: 'single_year' },
    GOLF:   { currentSeason: `${currentYear}`, lastSeason: `${currentYear - 1}`, seasonFormat: 'single_year' },
    DEFAULT:{ currentSeason: `${currentYear}`, lastSeason: `${currentYear - 1}`, seasonFormat: 'single_year' },
  };

  const sportKey    = category.replace('Sports - ', '').toUpperCase();
  const seasonCtx   = sportSeasons[sportKey] ?? sportSeasons.DEFAULT;

  const temporalContext: TemporalCtx = {
    today: now.toISOString().split('T')[0],
    currentYear, currentMonth,
    ...seasonCtx,
    dateAnchor:   `Today is ${now.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })}`,
    seasonAnchor: `The most recent completed ${sportKey || 'sports'} season is ${seasonCtx.lastSeason}. Current/ongoing season is ${seasonCtx.currentSeason}.`,
  };

  // Format config
  const slidesPerEntity = slidesPerEntityRaw.startsWith('2') ? 2 : 1;
  const entityCount     = Math.floor(slideCount / slidesPerEntity);

  let continuationStyle = 'same_name';
  if (/\(cont\.?\)/i.test(userContext))     continuationStyle = 'cont';
  else if (/continued/i.test(userContext))  continuationStyle = 'continued';
  else if (/part\s*2/i.test(userContext))   continuationStyle = 'part2';

  const formatConfig: FormatConfig = { slidesPerEntity, entityCount, isMultiSlideFormat: slidesPerEntity > 1, continuationStyle };

  // Title analysis
  const numberMatch  = title.match(/(\d+)\s+/);
  const promisedCount = numberMatch ? parseInt(numberMatch[1]) : entityCount;
  const isRanking    = /\b(top|best|greatest|worst|most|ranked|ranking)\b/i.test(title);
  const isListicle   = /\b(\d+)\s+(things?|ways?|reasons?|facts?|moments?|players?|movies?|shows?|athletes?|teams?)/i.test(title);
  const isTimeBased  = /\b(history|all[- ]time|ever|classic|legendary|iconic|memorable)\b/i.test(title);
  const emotionMatch = title.match(/\b(shocking|surprising|unbelievable|amazing|incredible|heartbreaking|hilarious|controversial|unexpected|memorable|iconic|legendary)\b/i);

  const colonSplit     = title.split(/[:–—-]/);
  const mainAngle      = colonSplit[0].trim();
  const secondaryAngle = colonSplit.length > 1 ? colonSplit.slice(1).join(' ').trim() : null;
  const requiresCorrelation = /mock draft|fit\s+(?:with|for)|compare|vs\.?|versus|how\s+\w+\s+(?:helps?|improves?)/i.test(title);

  const titleAnalysis: TitleAnalysis = {
    promisedCount, isRanking, isListicle, isTimeBased,
    emotionalPromise: emotionMatch ? emotionMatch[1].toLowerCase() : null,
    mainAngle, secondaryAngle, requiresCorrelation,
  };

  // Source parsing
  const allUrls = sourcesRaw
    .split(/[\n,]+/)
    .map(s => s.trim())
    .filter(s => s.startsWith('http'))
    .slice(0, 3);

  const userPrimaryUrl      = allUrls[0] ?? '';
  const userSecondaryUrls   = allUrls.slice(1);
  const isUserUrlRestricted = userPrimaryUrl ? isRestricted(userPrimaryUrl) : false;
  const hasValidUserSource  = !!userPrimaryUrl && !isUserUrlRestricted;
  const mustIncludeItems    = mustIncludeRaw ? mustIncludeRaw.split(/[\n,]+/).map(s => s.trim()).filter(Boolean) : [];

  return {
    title, category, slideCount, writerName, userContext, writingStyle,
    userPrimaryUrl, userSecondaryUrls, hasValidUserSource, isUserUrlRestricted,
    restrictedDomains: RESTRICTED_DOMAINS, mustIncludeItems,
    hasMustInclude: mustIncludeItems.length > 0,
    temporalContext, formatConfig, titleAnalysis,
    sourceCount: allUrls.filter(u => !isRestricted(u)).length,
    timestamp: new Date().toISOString(),
    isSports: category.startsWith('Sports'),
  };
}

// ── 2. firecrawlScrape (n8n: Firecrawl nodes) ────────────────────────────────

export async function firecrawlScrape(url: string, onlyMainContent = true): Promise<string> {
  const skipDomains = [
    'twitter.com','x.com','reddit.com','facebook.com','instagram.com','tiktok.com',
    'pinterest.com','quora.com','youtube.com','youtu.be','nytimes.com','wsj.com',
    'bloomberg.com','theathletic.com','linkedin.com',
  ];
  if (!url || !url.startsWith('http') || skipDomains.some(d => url.toLowerCase().includes(d))) {
    return '';
  }

  try {
    const resp = await axios.post(
      'https://api.firecrawl.dev/v2/scrape',
      {
        url,
        formats: ['markdown'],
        onlyMainContent,
        excludeTags: ['nav','footer','aside','script','style','header','ads','comments'],
      },
      { headers: { Authorization: `Bearer ${FIRECRAWL_KEY}` }, timeout: 60_000 },
    );
    return resp.data?.data?.markdown ?? resp.data?.markdown ?? '';
  } catch {
    return '';
  }
}

// ── 3. analyzeSourceAlignment (n8n: "Analyze Source Alignment") ──────────────

export async function analyzeSourceAlignment(
  prepData: PreparedData,
  primaryMarkdown: string,
  secondaryMarkdown: string,
  thirdMarkdown = '',
): Promise<SourcedData> {
  const allSourceContent = [
    primaryMarkdown  ? `=== PRIMARY SOURCE: ${prepData.userPrimaryUrl} ===\n${primaryMarkdown}`  : '',
    secondaryMarkdown ? `=== SECONDARY SOURCE: ${prepData.userSecondaryUrls[0] ?? ''} ===\n${secondaryMarkdown}` : '',
    thirdMarkdown ? `=== TERTIARY SOURCE: ${prepData.userSecondaryUrls[1] ?? ''} ===\n${thirdMarkdown}` : '',
  ].filter(Boolean).join('\n\n');

  if (!allSourceContent || allSourceContent.length < 100) {
    return { ...prepData, sourceAnalysis: { status: 'SCRAPE_FAILED', alignmentScore: 0, recommendation: 'SEARCH_FOR_SOURCES', scrapedContent: '', sourceCount: 0 } };
  }

  const sourceLower     = allSourceContent.toLowerCase();
  const titleWords      = prepData.title.toLowerCase().split(/\s+/).filter(w => w.length > 3);
  const keywordMatches  = titleWords.filter(w => sourceLower.includes(w));
  const keywordMatchRate = titleWords.length > 0 ? keywordMatches.length / titleWords.length : 0;

  const listPatterns = [/^\d+[.):\s]+/gm, /^#+\s+\d+/gm, /^\*\*\d+/gm];
  let estimatedItems = 0;
  for (const p of listPatterns) {
    const m = allSourceContent.match(p);
    if (m && m.length > estimatedItems) estimatedItems = m.length;
  }

  const factTypeChecks = {
    has_stats:  /\d+(?:\.\d+)?(?:\s*%|\s+yards?|\s+points?|\s+goals?)/i.test(allSourceContent),
    has_dates:  /\b(19|20)\d{2}\b/.test(allSourceContent),
    has_names:  /\b[A-Z][a-z]+\s+[A-Z][a-z]+\b/.test(allSourceContent),
  };

  const multiSourceBonus = prepData.sourceCount > 1 ? 10 : 0;
  const alignmentScore   = Math.min(100, Math.round(
    (keywordMatchRate * 40)
    + (Math.min(estimatedItems / prepData.formatConfig.entityCount, 1) * 30)
    + (factTypeChecks.has_stats ? 10 : 0)
    + (factTypeChecks.has_dates ? 10 : 0)
    + (factTypeChecks.has_names ? 10 : 0)
    + multiSourceBonus,
  ));

  let status: string, recommendation: string;
  if (alignmentScore >= 60)      { status = 'GOOD_MATCH';    recommendation = 'USE_SOURCE'; }
  else if (alignmentScore >= 35) { status = 'PARTIAL_MATCH'; recommendation = 'USE_WITH_SUPPLEMENTS'; }
  else                           { status = 'POOR_MATCH';    recommendation = 'SEARCH_FOR_SOURCES'; }

  let researchMode: string;
  if (alignmentScore >= 70 && estimatedItems >= prepData.formatConfig.entityCount * 0.9) {
    researchMode = 'CONTEXT_ONLY';
  } else if (alignmentScore >= 40) {
    researchMode = 'NORMAL_RESEARCH';
  } else {
    researchMode = 'DEEP_RESEARCH';
  }

  return {
    ...prepData,
    sourceAnalysis: {
      status, recommendation, alignmentScore, estimatedItems, factTypeChecks,
      scrapedContent: allSourceContent,
      sourceCount:   prepData.sourceCount,
      primaryLength: primaryMarkdown.length,
      secondaryLength: secondaryMarkdown.length,
      researchMode,
    },
  };
}

// ── 4. buildResearchStrategy (n8n: "Build Research Strategy") ────────────────

export async function buildResearchStrategy(prepData: PreparedData): Promise<SourcedData> {
  const authoritativeDomains: Record<string, string[]> = {
    'NFL':       ['espn.com','nfl.com','pro-football-reference.com'],
    'NBA':       ['espn.com','nba.com','basketball-reference.com'],
    'GOLF':      ['pga.com','golfdigest.com','golf.com'],
    'MOVIES & TV':['imdb.com','rottentomatoes.com','boxofficemojo.com'],
    'POP CULTURE':['people.com','eonline.com','imdb.com'],
    'DEFAULT':   ['espn.com','wikipedia.org'],
  };

  const sportKey       = prepData.category.replace('Sports - ', '').toUpperCase();
  const preferredDomains = authoritativeDomains[sportKey] ?? authoritativeDomains[prepData.category.toUpperCase()] ?? authoritativeDomains.DEFAULT;

  return {
    ...prepData,
    sourceAnalysis: { status: 'NO_SOURCE', recommendation: 'SEARCH_FOR_SOURCES', alignmentScore: 0, scrapedContent: '', researchMode: 'DEEP_RESEARCH' },
    preferredDomains,
  };
}

// ── 5. atomizeFacts (n8n: "Fact Atomizer") ───────────────────────────────────

export async function atomizeFacts(data: SourcedData): Promise<AtomizedData> {
  const sourceContent     = data.sourceAnalysis?.scrapedContent ?? '';
  const combinedRawContent = [sourceContent, data.userContext].filter(Boolean).join('\n\n');

  if (!combinedRawContent || combinedRawContent.length < 50) {
    return { ...data, atomizedFacts: [], factOnlyRepresentation: '', sourceSignatures: [], atomizationStats: { itemsProcessed: 0, totalFacts: 0 } };
  }

  const atomizedFacts: AtomizedFact[] = [];
  const sourceSignatures: string[]    = [];
  const sections = sourceContent ? sourceContent.split(/(?=##?\s*\d+[.):]?\s*)/) : [];

  sections.forEach((section, sectionIdx) => {
    if (section.trim().length < 30) return;
    const titleMatch = section.match(/^##?\s*(\d+)[.):]?\s*(.+?)(?:\n|$)/);
    const itemNumber = titleMatch ? parseInt(titleMatch[1]) : sectionIdx + 1;
    const itemName   = titleMatch ? titleMatch[2].trim().replace(/\*\*/g, '') : `Item ${sectionIdx + 1}`;
    const content    = section.replace(/^##?\s*\d+[.):]?\s*.+?\n/, '');
    const facts: AtomizedFact['facts'] = [];

    type StatDef = { pattern: RegExp; type: string };
    const statPatterns: StatDef[] = [
      { pattern: /(\d+(?:,\d{3})*(?:\.\d+)?)\s*(yards?|passing yards?|rushing yards?|touchdowns?|TDs?|points?|rebounds?|assists?)/gi, type: 'stat' },
      { pattern: /\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(million|billion|[MBK])?/gi, type: 'money' },
      { pattern: /(\d+(?:\.\d+)?)\s*%/g, type: 'percentage' },
      { pattern: /(\d+)\s*(Pro Bowls?|All-Stars?|MVP|Emmy|Oscar|Grammy|championships?|titles?|rings?)/gi, type: 'achievement' },
    ];

    for (const { pattern, type } of statPatterns) {
      const rx = new RegExp(pattern.source, pattern.flags);
      let m: RegExpExecArray | null;
      while ((m = rx.exec(content)) !== null) facts.push({ type, value: m[0].trim() });
    }

    for (const m of content.matchAll(/\b((19|20)\d{2})\b/g)) facts.push({ type: 'date', value: m[1] });
    for (const m of content.matchAll(/"([^"]{15,150})"/g))  facts.push({ type: 'quote', value: m[1], isExactQuote: true });

    const sentences = content.split(/[.!?]+/).filter(s => s.trim().length > 40);
    sentences.forEach(sentence => {
      const words = sentence.trim().split(/\s+/);
      for (let i = 0; i < words.length - 5; i++) {
        const phrase = words.slice(i, i + 6).join(' ').toLowerCase();
        if (!/\d.*\d.*\d/.test(phrase)) sourceSignatures.push(phrase);
      }
    });

    if (facts.length > 0 || content.length > 100) {
      atomizedFacts.push({ itemNumber, itemName, facts: facts.slice(0, 10), rawContent: content.slice(0, 400) });
    }
  });

  if (data.userContext && data.userContext.length > 50) {
    const contextFacts: AtomizedFact['facts'] = [];
    const ctxPatterns: Array<{ pattern: RegExp; type: string }> = [
      { pattern: /(\d+(?:,\d{3})*(?:\.\d+)?)\s*(yards?|passing yards?|rushing yards?|touchdowns?|TDs?|points?|rebounds?|assists?)/gi, type: 'stat' },
      { pattern: /\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(million|billion|[MBK])?/gi, type: 'money' },
      { pattern: /(\d+(?:\.\d+)?)\s*%/g, type: 'percentage' },
      { pattern: /(\d+)\s*(Pro Bowls?|All-Stars?|MVP|Emmy|Oscar|Grammy|championships?|titles?|rings?)/gi, type: 'achievement' },
    ];
    for (const { pattern, type } of ctxPatterns) {
      const rx = new RegExp(pattern.source, pattern.flags);
      let m: RegExpExecArray | null;
      while ((m = rx.exec(data.userContext)) !== null) contextFacts.push({ type, value: m[0].trim() });
    }
    for (const m of data.userContext.matchAll(/\b((19|20)\d{2})\b/g)) contextFacts.push({ type: 'date', value: m[1] });
    for (const m of data.userContext.matchAll(/"([^"]{15,150})"/g)) contextFacts.push({ type: 'quote', value: m[1], isExactQuote: true });
    if (contextFacts.length > 0) {
      atomizedFacts.push({ itemNumber: 0, itemName: 'USER_CONTEXT_DATA', facts: contextFacts.slice(0, 20), rawContent: data.userContext.slice(0, 800) });
    }
  }

  const factOnlyRepresentation = atomizedFacts.map(item => {
    const lines = [item.itemName === 'USER_CONTEXT_DATA' ? 'USER CONTEXT DATA:' : `ITEM ${item.itemNumber}: ${item.itemName}`];
    const stats        = item.facts.filter(f => ['stat','money','percentage'].includes(f.type));
    const achievements = item.facts.filter(f => f.type === 'achievement');
    const dates        = item.facts.filter(f => f.type === 'date');
    const quotes       = item.facts.filter(f => f.type === 'quote');
    if (stats.length)        lines.push(`  STATS: ${stats.map(s => s.value).join(', ')}`);
    if (achievements.length) lines.push(`  ACHIEVEMENTS: ${achievements.map(a => a.value).join(', ')}`);
    if (dates.length)        lines.push(`  DATES: ${[...new Set(dates.map(d => d.value))].join(', ')}`);
    if (quotes.length)       lines.push(`  QUOTES: ${quotes.map(q => `"${q.value}"`).join('; ')}`);
    return lines.join('\n');
  }).join('\n\n');

  return {
    ...data,
    atomizedFacts,
    factOnlyRepresentation,
    sourceSignatures: [...new Set(sourceSignatures)].slice(0, 150),
    atomizationStats: { itemsProcessed: atomizedFacts.length, totalFacts: atomizedFacts.reduce((s, i) => s + i.facts.length, 0) },
  };
}

// ── 6. Perplexity research (n8n: three Perplexity nodes) ─────────────────────

interface PerplexityRaw {
  choices: Array<{ message: { content: string } }>;
  citations?: string[];
}

async function callPerplexity(model: string, systemContent: string, userContent: string, maxTokens = 4000, timeoutMs = 120_000): Promise<PerplexityRaw> {
  const resp = await axios.post(
    'https://api.perplexity.ai/chat/completions',
    { model, messages: [{ role: 'system', content: systemContent }, { role: 'user', content: userContent }], max_tokens: maxTokens, return_citations: true, return_related_questions: false, temperature: 0.1 },
    { headers: { Authorization: `Bearer ${PERPLEXITY_KEY}`, 'Content-Type': 'application/json' }, timeout: timeoutMs },
  );
  return resp.data;
}

function buildPerplexitySystem(data: AtomizedData, deep: boolean): string {
  const tc = data.temporalContext;
  const loopingBlock = deep
    ? `\n\nRESEARCH STRATEGY & LOOPING:\nYou must search MULTIPLE times:\n1. First, search for the overall topic to get the full list\n2. Then search INDIVIDUALLY for each item to get detailed facts\n3. For items with thin data, run additional targeted searches\nDo NOT stop after one search. Keep searching until you have comprehensive data for ALL ${data.formatConfig.entityCount} items.`
    : '';
  const dateRules = !deep
    ? `\n\nCRITICAL DATE RULES:\n- 'Last season' means: ${tc.lastSeason}\n- 'Current season' means: ${tc.currentSeason}\n- Do NOT return data from ${tc.currentYear - 2} or earlier unless specifically historical\n- If current data unavailable, explicitly note the year of the data you're providing`
    : '';
  return `You are a senior research analyst acting as a completely automated backend data node. You have LIVE web search access.\n\n${tc.dateAnchor}\n${tc.seasonAnchor}\n\nSTRICT AUTOMATION RULES (CRITICAL):\n1. NO META-COMMENTARY: Never explain your search process, never apologize, never state what you cannot find, and never ask clarifying questions.\n2. NO CONVERSATIONAL FILLER: Do not output "Here is the research" or "I need clarification."\n3. AUTONOMOUS EXECUTION: If you cannot find a specific pre-existing list that perfectly matches the title, DO NOT STOP. You must autonomously identify, compile, and generate the ${data.formatConfig.entityCount} items yourself based on the overall topic.${loopingBlock}${dateRules}\n\nDATA QUALITY & INTEGRITY RULES:\n1. Never contradict the provided user content/existing facts.\n2. Always include exact numbers — no vague language.\n3. Prioritize: ESPN, official league/team sites, Billboard, IMDB, Wikipedia for verifiable facts.\n4. CITATION INTEGRITY:\n   - For quote-based facts: always attempt to find the PRIMARY source of each quote (book page, speech transcript, verified interview). If found only on aggregators, flag it as: "Primary origin: not verified".\n   - Only provide origin context if a source you actually searched EXPLICITLY states it.\n   - Every URL in your SOURCES must genuinely contain the specific fact cited. NEVER assign a source URL to a fact it does not actually contain.\n\nOUTPUT FORMAT FOR EACH ITEM:\n- Full name\n- 2-3 key stats with YEAR (must be from ${tc.lastSeason} season)\n- 1-2 achievements\n- One notable fact/quote\n- Source URL for EACH fact\n\nAt the top: PRIMARY SOURCE URL\nAt the bottom: Complete SOURCES list`;
}

function buildPerplexityUser(data: AtomizedData, isDeep: boolean): string {
  const tc = data.temporalContext;
  const mustBlock  = data.hasMustInclude
    ? (isDeep ? `MANDATORY ITEMS - research each one individually:\n` : `MANDATORY ITEMS (must include):\n`)
      + data.mustIncludeItems.map((m, i) => `${i + 1}. ${m}`).join('\n') + '\n\n'
    : '';
  const factsHeader = isDeep
    ? `EXISTING FACTS FROM USER SOURCE (DO NOT CONTRADICT. DO NOT ADD NEW STATS — provide background context only):\n`
    : `EXISTING FACTS FROM USER SOURCE (DO NOT CONTRADICT. DO NOT ADD NEW STATS — provide background context only):\n`;
  const factsBlock = data.factOnlyRepresentation
    ? factsHeader + data.factOnlyRepresentation.substring(0, isDeep ? 1500 : 2000) + '\n\n'
    : '';

  const intro = isDeep
    ? `DEEP RESEARCH REQUIRED for this MSN slideshow.`
    : `Research this MSN slideshow topic NOW using live web search.`;

  return `${intro}\n\nTitle: "${data.title}"\nCategory: ${data.category}\nItems needed: ${data.formatConfig.entityCount}${isDeep ? ' (comprehensive data for EACH)' : ''}\n\n${mustBlock}${factsBlock}SEARCH STRATEGY:\n1. Search: "${data.title} complete list"\n2. IF NO LIST IS FOUND: Autonomously identify ${data.formatConfig.entityCount} highly relevant items${isDeep ? ' that fit the Title and Category' : ''} and research them.\n3. For each item, search: "[item name] ${data.category} stats ${tc.lastSeason}"\n4. Search the live web to find additional hard numbers, exact dates, and supporting quotes.\n5. If any item has < 2 facts, search again${isDeep ? ' with different terms' : ''}.\n\nReturn EXACTLY ${data.formatConfig.entityCount} items with 3+ verified facts each.\nNEVER return items with only 1 fact - keep searching.\n\nSTART YOUR OUTPUT DIRECTLY WITH${isDeep ? ' THE' : ''} PRIMARY SOURCE URL${isDeep ? '' : ':'}. DO NOT OUTPUT ANY ${isDeep ? 'INTRODUCTORY OR CONCLUDING ' : ''}CONVERSATIONAL TEXT.`;
}

export async function perplexityDeepResearch(data: AtomizedData): Promise<PerplexityRaw> {
  return callPerplexity('sonar-pro', buildPerplexitySystem(data, true), buildPerplexityUser(data, true), 4000, 120_000);
}

export async function perplexityStandardResearch(data: AtomizedData): Promise<PerplexityRaw> {
  return callPerplexity('sonar', buildPerplexitySystem(data, false), buildPerplexityUser(data, false), 4000, 90_000);
}

export async function perplexityRetryResearch(data: AtomizedData): Promise<PerplexityRaw> {
  return callPerplexity('sonar-pro', buildPerplexitySystem(data, true), buildPerplexityUser(data, true), 4000, 120_000);
}

// ── 7. validateRetry (n8n: "Retry Validator") ────────────────────────────────

export async function validateRetry(data: AtomizedData, perplexityResp: PerplexityRaw): Promise<ResearchedData> {
  const answer    = perplexityResp?.choices?.[0]?.message?.content ?? '';
  const citations = perplexityResp?.citations ?? [];
  const wordCount = answer.split(/\s+/).filter(Boolean).length;

  const hardRefusals = ['cannot provide','knowledge was last updated','knowledge cutoff','unable to provide','has not taken place','future event'];
  const softRefusals = ['has not yet occurred','not yet been announced','fragmented and incomplete'];
  const lower = answer.toLowerCase();

  const hasHardRefusal = hardRefusals.some(p => lower.includes(p));
  const hasSoftRefusal = softRefusals.some(p => lower.includes(p));
  const needsRetry     = hasHardRefusal || wordCount < 150;

  return {
    ...data,
    perplexityAnswer:    answer,
    perplexityCitations: citations,
    perplexityWordCount: wordCount,
    needsRetry,
    retryReason: needsRetry ? (hasHardRefusal ? 'Perplexity refused the query' : `Only ${wordCount} words returned`) : null,
    hadSoftRefusal: hasSoftRefusal,
    relatedEsArticles: [],
  };
}

// ── 8. mergeResearch (n8n: "Merge Research") ─────────────────────────────────

export async function mergeResearch(data: ResearchedData, retryResp?: PerplexityRaw): Promise<MergedData> {
  let answer    = data.perplexityAnswer;
  let citations = data.perplexityCitations;

  if (retryResp) {
    const retryAnswer    = retryResp?.choices?.[0]?.message?.content ?? '';
    const retryCitations = retryResp?.citations ?? [];
    if (retryAnswer.length > answer.length) {
      answer    = retryAnswer;
      citations = retryCitations.length > citations.length ? retryCitations : citations;
    }
  }

  const primaryMatch      = answer.match(/PRIMARY SOURCE URL:\s*(https?:\/\/[^\s\n]+)/i);
  const perplexityPrimary = primaryMatch ? primaryMatch[1].trim() : '';
  const primarySourceUrl  = data.userPrimaryUrl || perplexityPrimary || citations[0] || '';

  const userSourceContent  = data.sourceAnalysis?.scrapedContent ?? '';
  const userContextContent = data.userContext ?? '';
  const alignScore         = data.sourceAnalysis?.alignmentScore ?? 0;

  let sourceQuality: string;
  if (userSourceContent.length > 2000 && alignScore >= 60) {
    sourceQuality = 'COMPREHENSIVE';
  } else if (userSourceContent.length > 500 || alignScore >= 35) {
    sourceQuality = 'PARTIAL';
  } else {
    sourceQuality = 'MINIMAL';
  }

  let combinedFactRepresentation = '';

  if (sourceQuality === 'COMPREHENSIVE') {
    combinedFactRepresentation += 'ABSOLUTE RULE: User source is COMPREHENSIVE.\nEVERY stat, name, ranking, date, quote MUST come from TIER 1A (scraped source) below.\nTIER 1B (User Context) contains ADDITIONAL facts, context, and instructions from the writer. USE these to enrich slides with extra information, angles, and details not in TIER 1A.\nTIER 2 (Perplexity) exists ONLY for tone/framing. ZERO facts from TIER 2.\nIf TIER 1A and TIER 2 conflict, TIER 1A wins. TIER 1B facts are authoritative and should be woven into slides.\n\n';
  } else if (sourceQuality === 'PARTIAL') {
    combinedFactRepresentation += 'PARTIAL USER SOURCE.\nTIER 1A (scraped source): Use for ALL items that appear in it.\nTIER 1B (User Context): USE to add information, facts, framing, and detail to ANY slide. Writer-provided data is authoritative.\nFor items NOT in TIER 1A or 1B, you may use TIER 2 Perplexity data but MARK with [P] tag.\n\n';
  } else {
    combinedFactRepresentation += 'NO SUBSTANTIAL SCRAPED SOURCE.\nTIER 1B (User Context) is your PRIMARY fact source if provided. Use all data from it.\nPerplexity research is your secondary data source. All TIER 2 data may also be used.\n\n';
  }

  if (userSourceContent || data.factOnlyRepresentation) {
    combinedFactRepresentation += '## TIER 1A: SCRAPED SOURCE — HIGHEST FACT AUTHORITY\nEvery stat, ranking, name, date, quote MUST come from here when available.\n\n';
    if (data.factOnlyRepresentation) {
      combinedFactRepresentation += '### Atomized Facts\n' + data.factOnlyRepresentation + '\n\n';
    }
    if (userSourceContent) {
      combinedFactRepresentation += '### Full Source Text\n' + userSourceContent + '\n\n';
    }
  }

  if (userContextContent) {
    combinedFactRepresentation += '## TIER 1B: USER CONTEXT — WRITER-PROVIDED DATA (USE TO ENRICH EVERY RELEVANT SLIDE)\nThis was provided directly by the writer. It contains facts, stats, angles, emphasis instructions, and context.\nRULES FOR TIER 1B:\n- Every fact, stat, or data point here is AUTHORITATIVE. Use it in the relevant slide.\n- If TIER 1B provides additional stats about an item already in TIER 1A, ADD those stats to the slide. Do not ignore them.\n- If TIER 1B mentions items, angles, or details not in TIER 1A, INCLUDE them in the article.\n- If TIER 1B contains instructions (tone, emphasis, angles), FOLLOW them as directives.\n- If TIER 1B directly contradicts a specific number in TIER 1A, TIER 1A wins. Otherwise TIER 1B stands as fact.\n- DO NOT discard TIER 1B content. The writer provided it for a reason.\n\n';
    combinedFactRepresentation += userContextContent + '\n\n';
  }

  if (sourceQuality === 'COMPREHENSIVE') {
    combinedFactRepresentation += '## TIER 2: PERPLEXITY — TONE & FRAMING ONLY (ZERO FACTS FROM HERE)\nDO NOT use any stat, date, ranking, achievement, or quote from this section.\nUse ONLY for: understanding why something matters, category context, writing tone.\n\n';
  } else if (sourceQuality === 'PARTIAL') {
    combinedFactRepresentation += '## TIER 2: PERPLEXITY — SUPPLEMENTARY (for items NOT in TIER 1A or 1B only)\nFor items that have TIER 1A or 1B data: IGNORE this section entirely for that item.\nFor items with NO TIER 1A or 1B data: you may use facts below, marked with [P].\n\n';
  } else {
    combinedFactRepresentation += '## TIER 2: PERPLEXITY RESEARCH — SECONDARY DATA SOURCE\nNo substantial scraped source provided. Use this alongside TIER 1B (if present) as your fact source.\n\n';
  }

  combinedFactRepresentation += answer.substring(0, 3000);

  return {
    ...data,
    perplexityAnswer:    answer,
    combinedFactRepresentation,
    primarySourceUrl,
    citations,
    sourceList: citations.map((url, i) => `[${i + 1}] ${url}`).join('\n'),
    researchWordCount: answer.split(/\s+/).length,
    researchOk: answer.length > 500 && citations.length >= 2,
    hasUserSource: !!(userSourceContent || data.factOnlyRepresentation),
    hasUserContext: !!userContextContent,
    sourceQuality,
    alignmentScore: alignScore,
  };
}

// ── 9. buildClaudePrompt (n8n: "Build Claude Prompt") ────────────────────────

export async function buildClaudePrompt(data: MergedData, citation1Markdown: string, citation2Markdown: string): Promise<PromptData> {
  function truncate(text: string, maxWords: number): string {
    const words = text.split(/\s+/);
    return words.length > maxWords ? words.slice(0, maxWords).join(' ') + '\n[truncated]' : text;
  }

  const citationContext = (data.sourceQuality === 'COMPREHENSIVE')
    ? ''
    : [
      citation1Markdown ? '\n\n=== TIER 2 BACKGROUND - SCRAPED CITATION 1 (CONTEXT ONLY, NOT A FACT SOURCE) ===\n' + truncate(citation1Markdown, 500) : '',
      citation2Markdown ? '\n\n=== TIER 2 BACKGROUND - SCRAPED CITATION 2 (CONTEXT ONLY, NOT A FACT SOURCE) ===\n' + truncate(citation2Markdown, 500) : '',
    ].filter(Boolean).join('');

  const BANNED_AI = `Delve, Embark, Foster, Navigate, Harness, Unlock, Elevate, Empower, Demystify, Catalyze, Optimize, Streamline, Tapestry, Landscape, Journey, Blueprint, Gateway, Intersection, Realm, Catalyst, Heartbeat, Pivotal, Comprehensive, Seamless, Vibrant, Dynamic, Synergistic, Multifaceted, Unparalleled, Robust, Transformative, Profound, Testament, Era, Synergy, "In today's world", "It is worth noting", "Moreover", "In conclusion", "Ultimately", "At the end of the day", "A testament to", "In today's fast-paced world", "In the rapidly evolving landscape of", "Since the dawn of", "Furthermore", "In addition to", "Conversely", "On the other hand", "Consequently", "It is important to note that", "In summary", "To wrap up", "As we look to the future", "A game-changer for", "The ultimate guide to", "Wait, There's More", "Therefore", "Hence", "Accordingly", "Nevertheless", "Nonetheless", "Despite this", "Critically important", "highly significant", "deeply impactful", "The future of [Topic] looks promising", "A wide variety of factors", "a plethora of options", "It serves as a testament to", "It acts as a catalyst for", showcase, underscore, highlight, cement, solidify, storied, remarkable, notable, impressive, outstanding, exceptional, incredible, unparalleled, unprecedented, larger than life, household name, the rest is history`;

  const BANNED_CONTENT = `Nude, Naked, Suicide, Kill, Shot, Stabbed, Fake News, Misinformation, Conspiracy Theory, Hoax, Exploitation, Fetish, Adultery, Scandal, Trans, War, Terrorist, shit, Vaccination, Weed, Cannabis, Murder, Prison, Fraud, Conspiracy, Jail, Racist, Sex, Sexual, Mutilate, Pussy, Vagina, Dick, Penis, Sexy, Fuck, Harassment, Marijuana, Cocaine, Assault, Scam, Gambling, Drug, Racism, Allegation, Vaccine, Ganja, Battery, Laundering, Butt, ass, Betting, Pedophile, Rape, Molest, Damn, Faggot, Fag, Nigga, Bitch, Cigarette, Cigar, Cum, Dominatrix, Ejaculation, Genitals, Hooters, Jackass, Masturbate, Nipple, NSFW, Onlyfans, Opioids, Orgasm, Pedos, Piss, Porn, Schlong, Smoking, Spunk, Striptease, Testicle, Tobacco, Vibrator, WTF`;

  const ta = data.titleAnalysis;
  const fc = data.formatConfig;
  const tc = data.temporalContext;
  const writingStyleBlock = data.writingStyle ? `\n\nWRITING STYLE INFLUENCE (from writer): ${data.writingStyle}\nApply this style naturally throughout the slideshow.` : '';

  const claudeSystemPrompt = `You are an expert MSN Slideshow writer for American audiences covering ${data.category}.

${tc.dateAnchor}
${tc.seasonAnchor}${writingStyleBlock}

SOURCE HIERARCHY — ABSOLUTE LAW (READ FIRST, OBEY ALWAYS)

You will receive content in three labeled tiers. The hierarchy is non-negotiable:

TIER 1A: SCRAPED SOURCE — HIGHEST FACT AUTHORITY
This is your primary fact source. Stats, names, rankings, dates, achievements, quotes, item ordering from here override everything else.

Rules for Tier 1A:
- If Tier 1A says "Player X had 42 TDs," write 42. Not "over 40," not "around 40," not "more than 40."
- If Tier 1A orders items 1-25, the article orders them 1-25. No re-ranking, no reordering.
- If Tier 1A lists 10 facts about an item, you may use any subset of those 10.
- If Tier 1A contradicts anything else, Tier 1A wins. Always.

TIER 1B: USER CONTEXT — WRITER-PROVIDED DATA (USE ACTIVELY)
The writer pasted this manually. It may contain additional facts, stats, angles, emphasis instructions, and context that MUST be used in the article.

CRITICAL — DATA vs. INSTRUCTIONS:
TIER 1B may contain BOTH data and instructions. You must distinguish between them:
- DATA (stats, names, dates, rankings, achievements, quotes) = use as FACTS in the relevant slides.
- INSTRUCTIONS (tone preferences, what to emphasize, things to avoid, formatting notes, angles to take) = follow as DIRECTIVES that shape how you write. Never quote an instruction as if it were a fact in a slide.

Rules for Tier 1B:
- Every fact, stat, or data point in Tier 1B is AUTHORITATIVE. Use it in the relevant slide.
- If Tier 1B provides additional stats about an item already in Tier 1A, ADD those stats to the slide. Do not ignore them.
- If Tier 1B mentions items, angles, or details not in Tier 1A, INCLUDE them in the article.
- If Tier 1B contains instructions (tone, what to emphasize, what to avoid), FOLLOW them as directives.
- If Tier 1B directly contradicts a specific number in Tier 1A, Tier 1A wins. For everything else, Tier 1B stands as fact.
- DO NOT discard Tier 1B content. The writer provided it for a reason.

TIER 2: PERPLEXITY RESEARCH + SCRAPED CITATIONS
This is BACKGROUND CONTEXT ONLY. NEVER a fact source.

Rules for Tier 2:
- Use ONLY for: tone calibration, historical framing, category background, understanding why something matters
- NEVER use for: stats, dates, rankings, achievements, quotes, specific factual claims
- If Tier 1A or 1B covers an item, ignore what Tier 2 says about that item
- If Tier 2 contradicts Tier 1A or 1B, Tier 2 is wrong by definition

TIER 3: NEUTRAL PUBLIC FACTS (LAST RESORT, MARK WITH [*])
Only for genuinely neutral facts not in Tier 1A or 1B that are common knowledge.

Allowed Tier 3: Team city, league name, standard role, sport's basic rules.
NEVER allowed as Tier 3: Any stat, career totals, championships, awards, years, dates, quotes, rankings.

Mark every Tier 3 use with [*] inline.

ANTI-HALLUCINATION PROTOCOL — NON-NEGOTIABLE

Before writing each slide, do this internal check:

STEP 1 — IDENTIFY: What is this slide about? Find the matching item in Tier 1A and Tier 1B.
STEP 2 — INVENTORY: List every fact Tier 1A gives you about this item, THEN every fact (not instruction) Tier 1B gives you. Combined, that is your complete fact pool.
STEP 3 — SELECT: Pick the 2-4 strongest facts from your inventory.
STEP 4 — WRITE: Build the slide using ONLY those selected facts plus connective tissue.
STEP 5 — VERIFY: Re-read the slide. For every specific claim ask: "Is this in my Tier 1A or Tier 1B fact inventory?" If YES: keep it. If NO: delete it. No exceptions.

WHEN TIER 1A IS THIN FOR AN ITEM:
- Check Tier 1B for additional facts about this item before writing a shorter slide
- If BOTH Tier 1A and 1B are thin, write a shorter, sharper slide using only what they provide
- A 38-word slide of pure truth beats a 48-word slide with one invented detail

WHAT YOU MUST NEVER DO:
- Invent a stat to round out a slide
- Pull a "well-known" career number from training when it's not in Tier 1A or 1B
- Add a championship, award, or milestone not mentioned in Tier 1A or 1B
- Quote anyone unless the quote is verbatim in Tier 1A or 1B
- Reorder, re-rank, or substitute items from Tier 1A's list
- Ignore Tier 1B data that the writer provided

ABSOLUTE OUTPUT RULE

You MUST always produce the complete slideshow. No exceptions.
These responses are FORBIDDEN:
- "I need more data before I can proceed"
- "The fact database is insufficient"
- Any response that is not the full formatted slideshow

ORIGINALITY: Facts are yours to use. Language is NOT.
Use any stat, date, name, achievement from Tier 1A or 1B. Write every sentence fresh.

TITLE-BODY CORRELATION (Highest Priority)

Title: "${data.title}"
- Numbers in title = exact count in body (${ta.promisedCount} items)
- Emotions (${ta.emotionalPromise ?? 'none detected'}) = explain WHO felt it, WHEN, WHY
- Main angle: ${ta.mainAngle}
${ta.secondaryAngle ? `- Secondary angle: ${ta.secondaryAngle}` : ''}
- If title makes a claim, literally substantiate it in the body using Tier 1A/1B facts
- Negative keywords in the title must find their place in the copy verbatim

WORD COUNTS (STRICT - Count every word)

- Meta Description: MAX 120 characters
- Intro slide (Slide 1): MAX 60 words
- Content slides: 35-50 words (aim for 40-45)
- If over or under, rewrite until it fits. Do not approximate.
- NEVER pad word count with invented facts. Use stronger writing instead.

META DESCRIPTION

Max 120 characters. Intriguing. Angle-focused. Has a hook.
Cannot be: CTA, reveal the main angle, paraphrased title.
AI patterns to NEVER use: "Discover the...", "Explore the top...", "Find out why...", "You won't want to miss..."
Good pattern: [Specific unexpected fact from Tier 1A or 1B]. [Implied question].

INTRO SLIDE (Slide 1) — MAX 60 WORDS

Your intro MUST:
- Create CURIOSITY — make readers NEED to scroll
- Include ONE surprising fact from Tier 1A or 1B tied to the theme
- Hint at what's coming WITHOUT naming specific items
- End with forward momentum
- Talk about something the title is promising

Your intro must NOT:
- Name any items from the list
- Reveal the #1 pick or any rankings
- Use "let's dive in" / "here are" / "we'll explore"
- Use generic openers ("Since the dawn of...", "In today's world...")
- Have generic background that assumes reader ignorance
- Use any fact not present in Tier 1A or 1B

WRITING VOICE — THE SPICY WRITER FACTOR

You are not summarizing facts. You are REACTING to them. Write like a sharp, witty sports columnist or pop culture critic who genuinely cares about the subject.

THE ENERGY RULES:
- Lead with what made YOU react. If a Tier 1A/1B stat shocked you, let that shock hit the reader first.
- One-sentence gut punches are your weapon.
- Contrast is your best friend. Set up expectation, then break it — using Tier 1A/1B facts.
- Specificity IS creativity.

RHYTHM AND PACING:
- Alternate sentence lengths deliberately. Short punch. Then longer context. Then short again.
- Never let two slides have the same energy.

EMOTIONAL TEXTURE (rotate through): DISBELIEF, RESPECT, HUMOR (light), TENSION, NOSTALGIA

WHAT TO AVOID:
- Wikipedia voice: "He is widely regarded as one of the greatest..."
- Cheerleader voice: "What an incredible, amazing, stunning performance!"
- Resume voice: stat-dumping without framing

CONTENT SLIDES — 5Ws + 1H Framework

Every slide must answer the RELEVANT questions using Tier 1A/1B:
WHO / WHAT / WHEN / WHERE / WHY / HOW

STATS NEED CONTEXT — MAX 2 STATS PER SLIDE

Every stat needs ONE of these as framing: WHY it matters / WHEN it happened / WHO it affected / WHAT it led to.

${fc.isMultiSlideFormat ? `MULTI-SLIDE FORMAT — 2 SLIDES PER ENTITY

SLIDE A: WHO they are, PRIMARY achievement (from Tier 1A/1B), key stat
SLIDE B: Supporting context, additional Tier 1A/1B stats, legacy/impact. Must BUILD ON Slide A using DIFFERENT facts.
` : ''}${ta.requiresCorrelation ? `CORRELATION WRITING

Every slide must CONNECT two things using Tier 1A/1B facts about both.
FORMULA: [Entity A's trait] + [How it addresses Entity B's need] + [Evidence]
` : ''}
QUALITY CONSISTENCY ENGINE

PRE-WRITING PLANNING (MANDATORY before writing Slide 1):
1. Read ALL Tier 1A and Tier 1B content completely
2. Separate Tier 1B into DATA (facts to use) and INSTRUCTIONS (directives to follow)
3. For EACH slide, identify the single strongest Tier 1A/1B fact that will anchor it
4. Check Tier 1B for additional facts that can enrich each slide
5. Verify slide 15's anchor is as specific as slide 3's
6. If ANY slide has no Tier 1A/1B anchor, write a shorter slide — do NOT pull from Tier 2 or training

THREE TESTS — every slide must pass ALL:
TEST 1 — STRANGER TEST: Reading only this slide, would someone learn one specific real thing?
TEST 2 — SIDE-BY-SIDE TEST: Is this as specific as slide 3?
TEST 3 — SOURCE TEST: Does every specific claim trace to Tier 1A or 1B?

NO FILLER SLIDES — ZERO TOLERANCE.

VARIETY ENFORCEMENT

1. OPENING WORDS — Never start 2 consecutive slides with the same word
2. SENTENCE STRUCTURE — Rotate through patterns A/B/C/D
3. ANTI-REPETITION: Track openings, structures, transitions, tones. Break patterns.

${ta.isRanking ? `RANKING ORDER — FOLLOW TIER 1A EXACTLY

If Tier 1A provides a ranked list, USE THAT EXACT ORDER. Do not re-rank.

For descending presentation:
Slide 2 = Tier 1A's rank ${ta.promisedCount} (lowest)
Last slide = Tier 1A's rank 1 (best/top)
Each slide title for rankings must start with the rank number.
` : ''}
HUMAN VOICE

- Clear, direct sentences. Vary length naturally.
- Let Tier 1A/1B facts create emotion — don't say "amazing", show the stat that IS amazing
- Sports lingo where appropriate
- Predominantly active voice
- No cliches, no forced regional metaphors

PUNCTUATION BANS (STRICT)

NO em-dashes (—). NO semicolons (;). NO ellipsis (...).
Banned in ALL slide descriptions. One violation = rewrite.

BANNED AI PHRASES — NEVER USE

${BANNED_AI}

BANNED CONTENT WORDS — NEVER USE

${BANNED_CONTENT}
Profanity in direct quotes only: censor as first letter + asterisks (s***, f***)
Dick cannot be used anywhere, including in names.

MSN SAFETY — 10-12 YEAR OLD TEST

Before every slide ask: "Should a 10-12 year old be reading this?"
Avoid: sexual content, graphic violence, drugs, gambling, political content, sensationalized celebrity drama, body shaming, bullying.
No profanity in titles or meta descriptions ever.

FORMAT (Plain text only, no markdown)

${data.title}

META: [Max 120 characters]

SLIDE 1
[Intro title]
[Max 60 words — Tier 1A/1B facts only]

SLIDE 2
[Creative title${ta.isRanking ? ' — start with rank number from Tier 1A' : ''}]
[35-50 words — Tier 1A/1B facts only]

...continue for all ${data.slideCount} slides...

SOURCES:
[URL]: [What Tier 1A/1B facts came from this source]`;

  const sourceQualityBlock = data.sourceQuality === 'COMPREHENSIVE'
    ? '\nSOURCE QUALITY: COMPREHENSIVE — Your scraped source covers all items.\nABSOLUTE RULE: Base facts on TIER 1A. Enrich with TIER 1B (user context data). Follow TIER 1B instructions. Zero facts from TIER 2.\nIf you cannot find a fact in TIER 1A or 1B, the article does not have that fact.\n'
    : data.sourceQuality === 'PARTIAL'
    ? '\nSOURCE QUALITY: PARTIAL — Your scraped source covers some items.\nFor items IN TIER 1A: use TIER 1A facts, enrich with TIER 1B data, follow TIER 1B instructions. For items NOT IN TIER 1A: use TIER 1B first, then TIER 2 marked with [P].\n'
    : '\nSOURCE QUALITY: MINIMAL — Use TIER 1B (user context) as your primary fact source if available. Follow any TIER 1B instructions. Use Perplexity research as secondary.\n';

  const claudeUserPrompt = `TIER 1A + 1B FACT DATABASE — YOUR ONLY FACT SOURCES

${data.combinedFactRepresentation}

${citationContext}

ASSIGNMENT

Title: "${data.title}"
Category: ${data.category}
Slides: 1 intro + ${data.slideCount} content slides
${fc.isMultiSlideFormat ? `Format: ${fc.slidesPerEntity} slides per entity (${fc.entityCount} entities total)` : ''}
${data.hasMustInclude ? `\nMANDATORY ITEMS (must all appear, all from Tier 1A/1B):\n${data.mustIncludeItems.map((m, i) => `${i + 1}. ${m}`).join('\n')}` : ''}

Primary Source: ${data.primarySourceUrl || 'See Tier 1A above'}

${sourceQualityBlock}
BEFORE WRITING — run this checklist:
1. What promise does the title make?
2. What is each slide's anchor fact from TIER 1A? (not Tier 2, not training)
3. What additional facts does TIER 1B provide for each slide? Have I planned to USE them?
4. Does TIER 1B contain any instructions? What are they? Am I prepared to FOLLOW them?
5. For items where Tier 1A is thin — does Tier 1B have data I should use before writing a shorter slide?
6. How will I vary structure across slides?
${ta.isRanking ? '7. Am I following Tier 1A\'s exact ranking order?' : ''}
${data.hasMustInclude ? `${ta.isRanking ? '8' : '7'}. Are ALL mandatory items covered using their Tier 1A/1B facts?` : ''}

CRITICAL REMINDERS:
- Every specific claim must trace to TIER 1A or TIER 1B
- TIER 1B (user context) is NOT optional — the writer provided it to be USED in the article
- TIER 1B data = use as facts. TIER 1B instructions = follow as directives. Never quote an instruction as slide content.
- Tier 2 (Perplexity, citations) is for tone/context only, NEVER facts
- If Tier 1A and 1B don't have a fact, the article doesn't have that fact
- Shorter true slides beat longer half-true slides
- Output the complete slideshow no matter what — never refuse, never ask for clarification

Write the complete slideshow now.`;

  return { ...data, claudeSystemPrompt, claudeUserPrompt };
}

// ── 10. generateWithClaude (n8n: "Claude - Generate Article") ─────────────────

export async function generateWithClaude(systemPrompt: string, userPrompt: string): Promise<unknown> {
  try {
    const resp = await axios.post(
      'https://api.anthropic.com/v1/messages',
      {
        model: 'claude-sonnet-4-5-20250929',
        max_tokens: 7500,
        temperature: 0.3,
        system: [{ type: 'text', text: systemPrompt, cache_control: { type: 'ephemeral' } }],
        messages: [{ role: 'user', content: userPrompt }],
      },
      {
        headers: {
          'Content-Type': 'application/json',
          'anthropic-version': '2023-06-01',
          'anthropic-beta': 'prompt-caching-2024-07-31',
          'x-api-key': ANTHROPIC_KEY,
        },
        timeout: 120_000,
      },
    );
    return resp.data;
  } catch (err: unknown) {
    if (axios.isAxiosError(err)) return err.response?.data ?? { type: 'error', error: { message: String(err.message) } };
    throw err;
  }
}

// ── 11. checkClaudeResponse (n8n: "Check Claude Response") ───────────────────

export async function checkClaudeResponse(claudeResp: unknown, promptData: PromptData): Promise<GeneratedData> {
  const resp = claudeResp as Record<string, unknown>;
  const contentArray = (resp?.content as Array<{ type: string; text: string }>) ?? [];
  const firstContent = contentArray[0] ?? {};

  let articleText = '';
  if (firstContent?.text) articleText = firstContent.text;
  else if (typeof resp?.content === 'string') articleText = resp.content as string;

  const isErrorResponse = resp?.type === 'error';
  const hasErrorObject  = !!resp?.error;
  const errorMsg        = ((resp?.error as Record<string, string>)?.message ?? '').toLowerCase();
  const isOverloaded    = (resp?.error as Record<string, string>)?.type === 'overloaded_error' || errorMsg.includes('overloaded');
  const isRateLimited   = (resp?.error as Record<string, string>)?.type === 'rate_limit_error' || errorMsg.includes('rate');
  const isCapacityError = errorMsg.includes('capacity');

  const hasValidContent = articleText.length > 200;
  const hasActualError  = isErrorResponse || hasErrorObject || isOverloaded || isRateLimited || isCapacityError;
  const claudeFailed    = hasActualError || !hasValidContent;

  return {
    ...promptData,
    articleText,
    originalArticleText: articleText,
    generatedBy:  claudeFailed ? '' : 'Claude',
    claudeFailed,
    failureReason: claudeFailed
      ? ((resp?.error as Record<string, string>)?.message || (resp?.error as Record<string, string>)?.type || (!hasValidContent ? `Response too short: ${articleText.length} chars` : 'Unknown error'))
      : null,
  };
}

// ── 12. generateWithGrok (n8n: "Grok - Generate Article (Fallback)") ──────────

export async function generateWithGrok(systemPrompt: string, userPrompt: string): Promise<string> {
  const resp = await axios.post(
    'https://api.x.ai/v1/chat/completions',
    { model: 'grok-4.20-0309-reasoning', max_tokens: 7000, temperature: 0.3, messages: [{ role: 'system', content: systemPrompt }, { role: 'user', content: userPrompt }] },
    { headers: { Authorization: `Bearer ${GROK_KEY}`, 'Content-Type': 'application/json' }, timeout: 300_000 },
  );
  return resp.data?.choices?.[0]?.message?.content ?? '';
}

// ── 13. validateStructure (n8n: "Structural Validator") ──────────────────────

export async function validateStructure(data: GeneratedData): Promise<ValidatedData> {
  let articleText = data.articleText;
  const errors: string[] = [], warnings: string[] = [], autoFixes: string[] = [];

  // Parse slides
  type SlidePartial = { slideNum: number; title: string; body: string; wordCount: number };
  const slides: SlidePartial[] = [];
  let currentSlide: number | null = null, currentTitle = '', currentBody = '';

  for (const line of articleText.split('\n')) {
    const trimmed    = line.trim().replace(/\*\*/g, '').trim();
    const slideMatch = trimmed.match(/^SLIDE\s*(\d+)/i);
    if (slideMatch) {
      if (currentSlide !== null) slides.push({ slideNum: currentSlide, title: currentTitle, body: currentBody.trim(), wordCount: currentBody.trim().split(/\s+/).filter(Boolean).length });
      currentSlide = parseInt(slideMatch[1]); currentTitle = ''; currentBody = '';
    } else if (currentSlide !== null && !currentTitle && trimmed && !trimmed.startsWith('META:')) {
      currentTitle = trimmed;
    } else if (currentSlide !== null && currentTitle && trimmed && !trimmed.startsWith('SOURCES')) {
      currentBody += ' ' + trimmed;
    }
  }
  if (currentSlide !== null) slides.push({ slideNum: currentSlide, title: currentTitle, body: currentBody.trim(), wordCount: currentBody.trim().split(/\s+/).filter(Boolean).length });

  // Word count
  slides.forEach(slide => {
    if (slide.slideNum === 1) {
      if (slide.wordCount > 60) errors.push(`Slide 1 (Intro): ${slide.wordCount} words – MAX is 60`);
    } else {
      if (slide.wordCount < 35) warnings.push(`Slide ${slide.slideNum}: ${slide.wordCount} words – min 35`);
      if (slide.wordCount > 50) warnings.push(`Slide ${slide.slideNum}: ${slide.wordCount} words – max 50`);
    }
  });

  const contentSlides = slides.filter(s => s.slideNum > 1).length;
  if (contentSlides !== data.slideCount) warnings.push(`Expected ${data.slideCount} content slides, found ${contentSlides}`);

  // Meta
  const metaMatch = articleText.match(/META:\s*([^\n]+)/i);
  const metaText  = metaMatch ? metaMatch[1].trim() : '';
  if (!metaText) {
    errors.push('No META description found');
  } else {
    if (metaText.length > 120) warnings.push(`Meta too long: ${metaText.length} chars (max 120)`);
    if (/find out|discover|see which|here are|we rank|you won't believe|explore/i.test(metaText)) warnings.push('Meta uses AI-generated patterns');
  }

  // Banned phrases
  const bannedPhrases = ['delve','embark','foster','navigate','harness','unlock','elevate','empower','tapestry','landscape','journey','blueprint','pivotal','comprehensive','seamless','vibrant','dynamic','robust','transformative','profound','moreover','furthermore','in conclusion','ultimately','game-changer','showcase','underscore','highlight','cement','solidify','storied','remarkable','notable','impressive','outstanding','exceptional','incredible','unparalleled','unprecedented'];
  const lowerArticle  = articleText.toLowerCase();
  const foundBanned   = bannedPhrases.filter(p => lowerArticle.includes(p));
  if (foundBanned.length) {
    warnings.push(`Banned phrases: ${foundBanned.join(', ')}`);
    ['moreover','furthermore','ultimately','in conclusion'].forEach(p => {
      if (lowerArticle.includes(p)) { articleText = articleText.replace(new RegExp(p, 'gi'), ''); autoFixes.push(`Removed "${p}"`); }
    });
  }

  // Punctuation bans
  slides.filter(s => s.slideNum > 0).forEach(slide => {
    if (slide.body.includes('—')) warnings.push(`Slide ${slide.slideNum}: em-dash (banned)`);
    if (slide.body.includes(';')) warnings.push(`Slide ${slide.slideNum}: semicolon (banned)`);
    if (slide.body.includes('...')) warnings.push(`Slide ${slide.slideNum}: ellipsis (banned)`);
  });

  // Unsafe content
  const unsafeWords = ['nude','suicide','sex','sexual','harassment','cocaine','marijuana','assault','rape','porn'];
  const foundUnsafe = unsafeWords.filter(w => lowerArticle.includes(w));
  if (foundUnsafe.length) errors.push(`UNSAFE content: ${foundUnsafe.join(', ')}`);

  // Quality degradation
  const contentSlideArr = slides.filter(s => s.slideNum > 1);
  if (contentSlideArr.length >= 10) {
    const third     = Math.ceil(contentSlideArr.length / 3);
    const avgFirst  = contentSlideArr.slice(0, third).reduce((s, x) => s + x.wordCount, 0) / third;
    const avgLast   = contentSlideArr.slice(-third).reduce((s, x) => s + x.wordCount, 0) / third;
    if (avgLast < avgFirst * 0.8) warnings.push(`Quality degradation: last third avg ${Math.round(avgLast)}w vs first third ${Math.round(avgFirst)}w`);
  }

  // Repetitive openings
  const openingWords = contentSlideArr.map(s => s.body.split(/\s+/)[0]?.toLowerCase()).filter(Boolean);
  const wordCounts: Record<string, number> = {};
  openingWords.forEach(w => wordCounts[w] = (wordCounts[w] ?? 0) + 1);
  const overused = Object.entries(wordCounts).filter(([, c]) => c > 2).map(([w, c]) => `"${w}" (${c}x)`);
  if (overused.length) warnings.push(`Repetitive openings: ${overused.join(', ')}`);

  // Plagiarism
  const sigs     = data.sourceSignatures ?? [];
  let plagScore  = 0;
  const plagMatches: string[] = [];
  if (sigs.length > 0) {
    const artWords = lowerArticle.split(/\s+/);
    for (let i = 0; i < artWords.length - 5; i++) {
      const phrase = artWords.slice(i, i + 6).join(' ');
      if (sigs.includes(phrase)) { plagMatches.push(phrase); plagScore += 5; }
    }
  }
  if (plagScore > 20) warnings.push(`Potential plagiarism: ${plagMatches.length} phrase matches`);

  // Intro spoiler check — intro should not name list items
  const introSlide = slides.find(s => s.slideNum === 1);
  if (introSlide) {
    const introText = introSlide.body.toLowerCase();
    const contentTitles = contentSlideArr.map(s => s.title.toLowerCase().replace(/^\d+\.\s*/, '').split(/\s+/).slice(0, 3).join(' ')).filter(t => t.length > 5);
    const spoilerFound = contentTitles.some(t => introText.includes(t));
    if (spoilerFound) warnings.push('Intro slide reveals list items (spoiler)');
  }

  // Stat provenance — check how many article stats appear in the user source
  const userSourceText = (data.sourceAnalysis?.scrapedContent ?? '') + '\n' + (data.userContext ?? '');
  const userSourceLower = userSourceText.toLowerCase().replace(/,/g, '');
  const articleStatsRaw = [...articleText.matchAll(
    /(\d+(?:,\d{3})*(?:\.\d+)?)\s*(yards?|points?|touchdowns?|TDs?|rebounds?|assists?|%|championships?|titles?|PPG|RPG|APG|MVPs?|Pro Bowls?|wins?|losses|steals?|blocks?|home runs?|RBIs?|ERA|batting average|goals?|saves?)/gi,
  )];
  const verifiedStats: string[] = [];
  const unverifiedStats: string[] = [];
  articleStatsRaw.forEach(match => {
    const statValue = match[1].replace(/,/g, '');
    const fullStat = match[0].trim();
    if (userSourceLower.includes(statValue)) verifiedStats.push(fullStat);
    else unverifiedStats.push(fullStat);
  });
  const provenanceRate = articleStatsRaw.length > 0
    ? Math.round((verifiedStats.length / articleStatsRaw.length) * 100) : 100;
  if (unverifiedStats.length > 0) {
    warnings.push(`Stats not in user source (possible hallucination): ${unverifiedStats.slice(0, 8).join(', ')}`);
  }
  if (provenanceRate < 70 && articleStatsRaw.length >= 5) {
    errors.push(`Low fact provenance: only ${provenanceRate}% of stats found in user source`);
  }

  // Stat-heavy without context
  contentSlideArr.forEach(slide => {
    const statCount = (slide.body.match(/\d+(?:,\d{3})*(?:\.\d+)?(?:\s*%|\s+yards?|\s+points?|\s+touchdowns?)?/g) || []).length;
    const hasContext = /because|after|when|during|meant|showed|proved|first time|only player|since|making|which|led to/i.test(slide.body);
    if (statCount >= 3 && !hasContext) warnings.push(`Slide ${slide.slideNum}: ${statCount} stats but lacks context`);
  });

  // Must-include coverage check
  if (data.mustIncludeItems && data.mustIncludeItems.length > 0) {
    const missing = data.mustIncludeItems.filter(item => !lowerArticle.includes(item.toLowerCase()));
    if (missing.length > 0) errors.push(`Must-include items missing: ${missing.join(', ')}`);
  }

  // Ranking order check — for ranking articles, slide 2 should be lowest rank
  if (data.titleAnalysis?.isRanking && contentSlideArr.length >= 3) {
    const firstTitle  = contentSlideArr[0]?.title ?? '';
    const lastTitle   = contentSlideArr[contentSlideArr.length - 1]?.title ?? '';
    const firstNum    = firstTitle.match(/^(\d+)/)?.[1];
    const lastNum     = lastTitle.match(/^(\d+)/)?.[1];
    if (firstNum && lastNum && parseInt(firstNum) < parseInt(lastNum)) {
      warnings.push(`Ranking order may be ascending — slide 2 should start at highest rank number (lowest position)`);
    }
  }

  const validationStatus = errors.length > 0 ? 'FAILED' : (warnings.length > 0 ? 'WARNINGS' : 'PASSED');

  return {
    ...data,
    articleText,
    structuralValidation: { status: validationStatus, errors, warnings, autoFixes, slideCount: slides.length, metaLength: metaText.length },
    plagiarismCheck:      { score: plagScore, matches: plagMatches.slice(0, 10), status: plagScore > 30 ? 'HIGH' : plagScore > 10 ? 'MEDIUM' : 'LOW' },
    factProvenance:       { rate: provenanceRate, verified: verifiedStats.length, unverified: unverifiedStats.length, total: articleStatsRaw.length, unverifiedExamples: unverifiedStats.slice(0, 10) },
    slides,
  };
}

// ── 14. extractClaims (n8n: "Extract Claims") ────────────────────────────────

export async function extractClaims(data: ValidatedData): Promise<ClaimedData> {
  const articleText = data.articleText;
  const claims: ClaimedData['claimsToVerify'] = [];

  for (const m of articleText.matchAll(/(\d+(?:,\d{3})*(?:\.\d+)?)\s*(yards?|points?|touchdowns?|TDs?|rebounds?|assists?|wins?|%|million|billion|championships?|titles?|Pro Bowls?|MVPs?|Emmy|Oscar)/gi)) {
    const start = Math.max(0, m.index! - 40);
    const end   = Math.min(articleText.length, m.index! + m[0].length + 40);
    claims.push({ type: 'stat', claim: m[0], context: articleText.slice(start, end).replace(/\n/g, ' ') });
  }
  for (const m of articleText.matchAll(/\b((19|20)\d{2})\b/g)) {
    const start = Math.max(0, m.index! - 40);
    const end   = Math.min(articleText.length, m.index! + 4 + 40);
    claims.push({ type: 'date', claim: m[1], context: articleText.slice(start, end).replace(/\n/g, ' ') });
  }
  for (const m of articleText.matchAll(/\b(first|only|most|best|worst|largest|oldest|youngest|fastest|highest|record|all-time)\b[^.]{10,80}/gi)) {
    claims.push({ type: 'superlative', claim: m[0].trim(), context: m[0].trim() });
  }

  const seen = new Set<string>();
  const uniqueClaims = claims.filter(c => { const k = c.claim.toLowerCase(); if (seen.has(k)) return false; seen.add(k); return true; });

  return { ...data, claimsToVerify: uniqueClaims.slice(0, 25) };
}

// ── 15. grokFactCheck (n8n: "Grok - Fact Check") ─────────────────────────────

export async function grokFactCheck(data: ClaimedData): Promise<unknown> {
  const tc = data.temporalContext;
  const systemContent = `You are a STRICT fact-checker with LIVE web search access. Your ONLY job is to verify every factual claim in this MSN slideshow article. You do NOT rewrite, edit, or audit formatting. ONLY verify facts.\n\n${tc.dateAnchor}\n${tc.seasonAnchor}\n\nPay special attention to:\n- Stats that seem rounded or approximated\n- Dates and years (last season = ${tc.lastSeason}, current season = ${tc.currentSeason})\n- Rankings, superlatives, and record claims\n- Quotes — verify exact wording and attribution\n- Player/team associations and transfers\n- Awards and achievements\n\nVERIFICATION PROTOCOL\n\nFor EACH slide in the article:\n1. IDENTIFY every specific factual claim (stats, dates, rankings, awards, records, superlatives, quotes, team/player associations)\n2. SEARCH the web for each claim independently — do NOT rely on training data\n3. COMPARE what the article says vs what authoritative sources say\n4. FLAG any discrepancy, no matter how small\n\nVERIFICATION STANDARDS:\n- A stat is VERIFIED only if you find the EXACT number in a reliable source\n- A stat is INCORRECT if your sources show a different number\n- A stat is UNVERIFIABLE if no reliable source confirms or denies it after searching\n- "Close enough" is NOT verified. 18.0 PPG is not 17.9 PPG. 42 TDs is not 41 TDs.\n- Rounded numbers must be flagged\n- Rankings: verify the exact ranking number, not just "top 25"\n\nSOURCE PRIORITY:\nESPN, official league/conference sites, Sports Reference family, team official sites, AP/Reuters > Wikipedia > CBS Sports, Fox Sports > aggregator blogs\n\nOUTPUT FORMAT — FOLLOW THIS EXACTLY\n\n--- SLIDE [N]: [Entity Name] ---\nCLAIM 1: "[exact claim text from article]"\n  STATUS: VERIFIED | INCORRECT | UNVERIFIABLE\n  FOUND: [what your web search actually found]\n  SOURCE: [URL]\n  CORRECTION: [only if INCORRECT]\n\n...continue for ALL claims in ALL slides...\n\n--- VERIFICATION SUMMARY ---\nTotal claims checked: [N]\nVerified: [N]\nIncorrect: [N]\nUnverifiable: [N]\nVerification rate: [X]%\n\n--- INCORRECT CLAIMS LIST ---\nSlide [N]: "[wrong claim]" → SHOULD BE: "[corrected claim]" (Source: [URL])\n\nIf zero incorrect claims found, write: No incorrect claims detected.\n\nSTART OUTPUT NOW. Begin directly with --- SLIDE 1 ---`;
  const userContent = `Fact-check this MSN slideshow article. Search the web and verify EVERY specific claim — stats, dates, rankings, transfers, achievements, quotes.\n\nTitle: "${data.title}"\nCategory: ${data.category}\nLast completed season: ${tc.lastSeason}\nCurrent/ongoing season: ${tc.currentSeason}\n\nPrimary Source URL: ${data.primarySourceUrl || 'Not provided'}\n\nARTICLE TO VERIFY\n\n${data.articleText}\n\nVerify EVERY claim in EVERY slide. Search the web for each one. Do not skip slides. Do not assume anything is correct. Output in the exact format specified.`;

  const resp = await axios.post(
    'https://api.x.ai/v1/responses',
    {
      model: 'grok-4-fast-non-reasoning',
      tools: [{ type: 'web_search' }, { type: 'x_search' }],
      max_output_tokens: 7000,
      temperature: 0.0,
      prompt_cache_key: 'msn-slideshow-auditor-v7',
      input: [
        { role: 'system', content: systemContent },
        { role: 'user', content: userContent },
      ],
    },
    { headers: { Authorization: `Bearer ${GROK_KEY}`, 'Content-Type': 'application/json' }, timeout: 300_000 },
  );
  return resp.data;
}

// ── 16. processVerification (n8n: "Process Fact Check Results") ──────────────

export async function processVerification(data: ClaimedData, verifyResp: unknown): Promise<VerifiedData> {
  const resp = verifyResp as Record<string, unknown>;
  let text = '';

  // Grok /v1/responses format
  const outputs = (resp?.output as Array<Record<string, unknown>>) ?? [];
  if (outputs.length > 0) {
    const msgOutput = [...outputs].reverse().find((o: Record<string, unknown>) => o.type === 'message');
    if (msgOutput?.content) {
      const contentArr = msgOutput.content as Array<{ type: string; text?: string }>;
      text = contentArr.find(c => c.type === 'output_text')?.text ?? '';
    }
  }

  // Fallback to /chat/completions format
  if (!text) {
    const plexResp = resp as { choices?: Array<{ message: { content: string } }> };
    text = plexResp?.choices?.[0]?.message?.content ?? '';
  }

  // Extract URLs from text
  const urlMatches = [...text.matchAll(/https?:\/\/[^\s)>\"\]]+/g)];
  const citations = [...new Set(urlMatches.map(m => m[0]))].slice(0, 10);

  // Parse n8n-style slide-by-slide verification output
  let articleText = data.articleText;
  const results: VerifiedData['perplexityVerification']['results'] = [];
  const claimPattern = /CLAIM\s*\d+:\s*"([^"]+)"\s*\n\s*STATUS:\s*(VERIFIED|INCORRECT|UNVERIFIABLE)\s*\n\s*FOUND:\s*([^\n]+)\s*\n\s*SOURCE:\s*([^\n]+)(?:\s*\n\s*CORRECTION:\s*([^\n]+))?/gi;
  let m: RegExpExecArray | null;
  let claimIdx = 0;
  while ((m = claimPattern.exec(text)) !== null) {
    const claim = m[1].trim();
    const status = m[2].toUpperCase();
    const found = m[3].trim();
    const source = m[4].trim();
    const correction = m[5]?.trim() ?? null;
    results.push({ claimIndex: claimIdx++, claim, status, finding: found, source });

    // Auto-correct numeric facts marked INCORRECT when a correction is provided
    if (status === 'INCORRECT' && correction) {
      const claimNums = claim.match(/[\d,]+(?:\.\d+)?/g);
      const corrNums = correction.match(/[\d,]+(?:\.\d+)?/g);
      if (claimNums && corrNums && claimNums[0] && corrNums[0]) {
        articleText = articleText.replace(claimNums[0], corrNums[0]);
      }
    }
  }

  const verified     = results.filter(r => r.status === 'VERIFIED').length;
  const incorrect    = results.filter(r => r.status === 'INCORRECT').length;
  const unverifiable = results.filter(r => r.status === 'UNVERIFIABLE').length;

  return {
    ...data,
    articleText,
    originalArticleText: data.articleText,
    perplexityVerification: {
      results, citations,
      stats: { verified, incorrect, unverifiable, total: results.length },
      score: results.length > 0 ? Math.round((verified / results.length) * 100) : 0,
    },
  };
}

// ── 17. grokAuditAndVerify (n8n: "Grok - Audit & Verify") ────────────────────

export async function grokAuditAndVerify(data: VerifiedData): Promise<string> {
  const tc = data.temporalContext;
  const ta = data.titleAnalysis;
  const systemContent = `You are an MSN Slideshow compliance auditor. Fact-checking has ALREADY been done — do NOT fact-check. Your ONLY job is to audit the article against editorial rules and MSN content safety standards, fix violations you can fix, and flag the rest.

${tc.dateAnchor}

AUDIT CHECKLIST — CHECK EVERY SINGLE RULE

1. STRUCTURE
- Meta description exists and is max 120 characters?
- Meta is NOT a CTA, does NOT paraphrase the title, no AI patterns ("Discover the...", "Explore the...", "Find out why...")?
- Intro slide (Slide 1): Max 60 words?
- Every content slide: 35-50 words? (Count carefully — flag any slide outside this range with exact word count)
- Correct total number of content slides?
- If ranking: slide titles start with the rank number?

2. TITLE-BODY CORRELATION
- Every promise made in the title is delivered in the body?
- If title has a number, does the article have exactly that many?
- If title has emotional words, does the body explain WHO felt it, WHEN, WHY?
- Negative keywords in the title appear verbatim somewhere in the copy?

3. INTRO QUALITY
- Intro has at least ONE specific fact tied to the theme (not generic filler)?
- Intro does NOT name any specific items from the list?
- Intro does NOT reveal rankings or the #1 pick?
- No generic openers: "Since the dawn of", "In today's world", "When it comes to"?
- No "let's dive in", "here are", "we'll explore"?
- Intro ends with forward momentum?

4. PUNCTUATION BANS — ZERO TOLERANCE
- No em-dashes anywhere in slide descriptions?
- No semicolons anywhere in slide descriptions?
- No ellipsis anywhere in slide descriptions?
If found: REMOVE THEM in your corrected output.

5. BANNED PHRASES — ZERO TOLERANCE
If ANY of these appear anywhere in the article, REMOVE THEM:
Delve, Embark, Foster, Navigate, Harness, Unlock, Elevate, Empower, Demystify, Catalyze, Optimize, Streamline, Tapestry, Landscape, Journey, Blueprint, Gateway, Intersection, Realm, Catalyst, Heartbeat, Pivotal, Comprehensive, Seamless, Vibrant, Dynamic, Synergistic, Multifaceted, Unparalleled, Robust, Transformative, Profound, Testament, Era, Synergy, Moreover, Furthermore, In conclusion, Ultimately, At the end of the day, A testament to, In today's fast-paced world, Since the dawn of, It is worth noting, Game-changer, showcase, underscore, highlight, cement, solidify, storied, remarkable, notable, impressive, outstanding, exceptional, incredible, unparalleled, unprecedented, larger than life, household name, the rest is history

6. BANNED CONTENT WORDS — HARD BLOCK
Nude, Naked, Suicide, Kill, Shot, Stabbed, Fake News, Conspiracy Theory, Exploitation, Fetish, Adultery, Scandal, Trans, War, Terrorist, shit, Vaccination, Weed, Cannabis, Murder, Prison, Fraud, Conspiracy, Jail, Racist, Sex, Sexual, Mutilate, Pussy, Vagina, Dick, Penis, Sexy, Fuck, Harassment, Marijuana, Cocaine, Assault, Scam, Gambling, Drug, Racism, Allegation, Vaccine, Damn, Bitch, Porn, NSFW
Note: "Dick" cannot be used ANYWHERE including in names.

7. WRITING QUALITY
- No robotic/Wikipedia phrasing?
- No cheerleader voice?
- Predominantly active voice?
- No two consecutive slides start with the same word?
- Sentence length varies across slides?
- No stat-dumping (3+ stats without context)?
- No slide is pure filler?

8. MSN SAFETY — 10-12 YEAR OLD TEST
- Every slide is safe for a 10-12 year old?
- No profanity in titles or meta description?

OUTPUT FORMAT — FOLLOW THIS EXACTLY

--- RULE COMPLIANCE ---

1. STRUCTURE: [PASS/FAIL]
   [If FAIL: list specific violations]

2. TITLE-BODY: [PASS/FAIL]
   [If FAIL: what promise was broken]

3. INTRO: [PASS/FAIL]
   [If FAIL: what's wrong]

4. PUNCTUATION: [PASS/FAIL]
   [If FAIL: list every violation]

5. BANNED PHRASES: [PASS/FOUND]
   [If FOUND: list every banned phrase and which slide]

6. BANNED CONTENT WORDS: [PASS/FOUND]
   [If FOUND: list every banned word and which slide]

7. WRITING QUALITY: [PASS/FAIL]
   [If FAIL: specific issues]

8. MSN SAFETY: [PASS/FAIL]
   [If FAIL: specific concerns]

--- CORRECTED ARTICLE ---
[Output the FULL article with all fixable violations corrected.
If NO corrections needed, write the original article unchanged.]

--- AUDIT SUMMARY ---
Rules passed: [X]/8
Violations found: [N]
Auto-corrections applied: [list what you fixed]
Flags for human review: [list what you couldn't auto-fix]
=== END AUDIT ===`;

  const resp = await axios.post(
    'https://api.x.ai/v1/chat/completions',
    {
      model: 'grok-3-latest', max_tokens: 6000, temperature: 0.0,
      messages: [
        { role: 'system', content: systemContent },
        { role: 'user',   content: `Audit this MSN slideshow against ALL editorial rules and content safety standards. Do NOT fact-check — that is already done.\n\nTitle: "${data.title}"\nCategory: ${data.category}\nExpected content slides: ${data.slideCount}\nIs ranking: ${ta?.isRanking ? 'YES — slide titles must start with rank number' : 'NO'}\nPromised count in title: ${ta?.promisedCount ?? 'N/A'}\nEmotional promise: ${ta?.emotionalPromise ?? 'None'}\n\nARTICLE TO AUDIT\n\n${data.articleText}\n\nCheck EVERY rule. Fix what you can. Flag what needs human review. Output the corrected article in full.` },
      ],
    },
    { headers: { Authorization: `Bearer ${GROK_KEY}`, 'Content-Type': 'application/json' }, timeout: 400_000 },
  );
  return resp.data?.choices?.[0]?.message?.content ?? '';
}

// ── 18. extractAuditResults (n8n: "Extract Audit Results") ───────────────────

export async function extractAuditResults(data: VerifiedData, grokText: string): Promise<AuditedData> {
  const originalArticle = data.articleText;

  if (!grokText || grokText.length < 200) {
    return {
      ...data,
      grokAudit: { status: 'FAILED', rawResponse: '', summary: '', stats: { rulesPassed: 'N/A', violations: 0, corrections: 'None', flags: 'None' } },
      grokSources: [], combinedSourceList: [], combinedSourceListText: '',
      rewriteApplied: false,
    };
  }

  // Parse n8n-format sections
  const correctionMatch = grokText.match(/--- CORRECTED ARTICLE ---\s*([\s\S]*?)(?=--- AUDIT SUMMARY|$)/i);
  const summaryMatch    = grokText.match(/--- AUDIT SUMMARY ---\s*([\s\S]*?)(?====\s*END AUDIT|$)/i);

  let correctedArticle = originalArticle;
  if (correctionMatch) {
    const correctionContent = correctionMatch[1].trim();
    if (correctionContent && correctionContent !== 'NO CORRECTIONS NEEDED' && correctionContent.length > 500) {
      correctedArticle = correctionContent;
    }
  }

  const auditSummary = summaryMatch?.[1]?.trim() ?? '';

  // Parse audit stats
  const passCount = (grokText.match(/: PASS/gi) || []).length;
  const failCount = (grokText.match(/: FAIL/gi) || []).length;
  const foundCount = (grokText.match(/: FOUND/gi) || []).length;

  const rulesPassedMatch = auditSummary.match(/Rules passed:\s*(\d+)\/(\d+)/i);
  const violationsMatch  = auditSummary.match(/Violations found:\s*(\d+)/i);
  const correctionsMatch = auditSummary.match(/Auto-corrections applied:\s*([^\n]+)/i);
  const flagsMatch       = auditSummary.match(/Flags for human review:\s*([^\n]+)/i);

  // Build combined source list from fact check citations
  const factCheckSources: SourceEntry[] = (data.perplexityVerification?.citations ?? []).map((url, i) => ({
    index: i + 1, url, verifiedBy: 'Grok Fact Check',
    factsVerified: (data.perplexityVerification?.results?.filter(r => r.source === url).map(r => r.claim).join(', ')) || 'General verification',
  }));

  const combinedSourceListText = factCheckSources.map((s, i) => `${i + 1}. ${s.url}\n   Verified by: ${s.verifiedBy}\n   Facts: ${s.factsVerified}`).join('\n\n');

  return {
    ...data,
    originalArticleText: originalArticle,
    articleText: correctedArticle,
    grokAudit: {
      status: 'COMPLETED',
      rawResponse: grokText,
      summary: auditSummary,
      stats: {
        rulesPassed: rulesPassedMatch ? `${rulesPassedMatch[1]}/${rulesPassedMatch[2]}` : `${passCount}/${passCount + failCount + foundCount}`,
        violations:  violationsMatch ? parseInt(violationsMatch[1]) : failCount + foundCount,
        corrections: correctionsMatch?.[1]?.trim() ?? 'None',
        flags:       flagsMatch?.[1]?.trim() ?? 'None',
      },
    },
    grokSources: [], combinedSourceList: factCheckSources, combinedSourceListText,
    rewriteApplied: correctedArticle !== originalArticle,
  };
}

// ── 19. finalAssembly (n8n: "Final Assembly") ────────────────────────────────

export async function finalAssembly(data: AuditedData): Promise<FinalOutput> {
  const researchScore    = data.researchOk ? 80 : 50;
  const factCheckScore   = data.perplexityVerification?.score ?? 50;
  const structuralScore  = data.structuralValidation?.status === 'PASSED' ? 100 : data.structuralValidation?.status === 'WARNINGS' ? 70 : 40;
  const plagiarismScore  = data.plagiarismCheck?.status === 'LOW' ? 100 : data.plagiarismCheck?.status === 'MEDIUM' ? 70 : 40;
  const qualityScore     = Math.round((researchScore * 0.15) + (factCheckScore * 0.40) + (structuralScore * 0.20) + (plagiarismScore * 0.25));

  const summaryParts: string[] = [];
  if (data.structuralValidation?.errors?.length)  summaryParts.push(`Errors: ${data.structuralValidation.errors.join('; ')}`);
  if (data.structuralValidation?.warnings?.length) summaryParts.push(`Warnings: ${data.structuralValidation.warnings.length}`);
  if (data.rewriteApplied)                         summaryParts.push('Grok corrections applied');
  if (data.grokAudit?.stats?.flags && data.grokAudit.stats.flags !== 'None') summaryParts.push(`Flags: ${data.grokAudit.stats.flags}`);
  if (data.generatedBy)                            summaryParts.push(`Generated by: ${data.generatedBy}`);

  const auditReport = `
QUALITY AUDIT REPORT

OVERALL QUALITY SCORE: ${qualityScore}/100

RESEARCH: ${researchScore}/100
FACT VERIFICATION (Grok): ${factCheckScore}/100
STRUCTURE: ${structuralScore}/100
ORIGINALITY: ${plagiarismScore}/100

GROK FACT CHECK
Verified: ${data.perplexityVerification?.stats?.verified ?? 0}
Incorrect: ${data.perplexityVerification?.stats?.incorrect ?? 0}
Unverifiable: ${data.perplexityVerification?.stats?.unverifiable ?? 0}
Total claims: ${data.perplexityVerification?.stats?.total ?? 0}

RULES AUDIT (Grok)
Rules Passed: ${data.grokAudit?.stats?.rulesPassed ?? 'N/A'}
Violations: ${data.grokAudit?.stats?.violations ?? 0}
Corrections: ${data.grokAudit?.stats?.corrections ?? 'None'}
Flags: ${data.grokAudit?.stats?.flags ?? 'None'}

STRUCTURAL VALIDATION
Status: ${data.structuralValidation?.status ?? 'N/A'}
Errors: ${data.structuralValidation?.errors?.join(', ') || 'None'}
Warnings: ${data.structuralValidation?.warnings?.join(', ') || 'None'}
`;

  return {
    title: data.title, category: data.category, slideCount: data.slideCount, writerName: data.writerName,
    articleText:         data.articleText,
    originalArticleText: data.originalArticleText || data.articleText,
    auditReport, qualityScore, researchScore,
    verificationScore: factCheckScore, structuralScore, originalityScore: plagiarismScore,
    primarySourceUrl:       data.primarySourceUrl,
    combinedSourceListText: data.combinedSourceListText,
    summaryComment:  summaryParts.length > 0 ? summaryParts.join('. ') : 'Article passed all checks.',
    validationStatus: data.structuralValidation?.status ?? 'UNKNOWN',
    factsVerified:     `${data.perplexityVerification?.stats?.verified ?? 0}/${data.perplexityVerification?.stats?.total ?? 0}`,
    grokRulesPassed: data.grokAudit?.stats?.rulesPassed ?? 'N/A',
    flagsForReview:    data.grokAudit?.stats?.flags ?? 'None',
    generatedBy:  data.generatedBy ?? 'Unknown',
    generatedAt:  new Date().toISOString(),
  };
}

