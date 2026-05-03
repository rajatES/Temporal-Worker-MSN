import axios from 'axios';
import * as dotenv from 'dotenv';
import {
  FormInput, PreparedData, SourcedData, AtomizedData, ResearchedData,
  MergedData, PromptData, GeneratedData, ValidatedData, ClaimedData,
  VerifiedData, AuditedData, FinalOutput, TemporalCtx,
  FormatConfig, TitleAnalysis, AtomizedFact, SourceEntry,
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
): Promise<SourcedData> {
  const allSourceContent = [
    primaryMarkdown  ? `=== PRIMARY SOURCE: ${prepData.userPrimaryUrl} ===\n${primaryMarkdown}`  : '',
    secondaryMarkdown ? `=== SECONDARY SOURCE: ${prepData.userSecondaryUrls[0] ?? ''} ===\n${secondaryMarkdown}` : '',
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

  return {
    ...prepData,
    sourceAnalysis: {
      status, recommendation, alignmentScore, estimatedItems, factTypeChecks,
      scrapedContent: allSourceContent,
      sourceCount:   prepData.sourceCount,
      primaryLength: primaryMarkdown.length,
      secondaryLength: secondaryMarkdown.length,
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
    sourceAnalysis: { status: 'NO_SOURCE', recommendation: 'SEARCH_FOR_SOURCES', alignmentScore: 0, scrapedContent: '' },
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

  const factOnlyRepresentation = atomizedFacts.map(item => {
    const lines = [`ITEM ${item.itemNumber}: ${item.itemName}`];
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

function buildPerplexitySystem(data: AtomizedData, _deep: boolean): string {
  const tc = data.temporalContext;
  return `You are a senior research analyst acting as a completely automated backend data node. You have LIVE web search access.\n\n${tc.dateAnchor}\n${tc.seasonAnchor}\n\nSTRICT AUTOMATION RULES:\n1. NO META-COMMENTARY. Never explain your search process or ask clarifying questions.\n2. AUTONOMOUS EXECUTION: If no pre-existing list matches the title, autonomously identify and compile the ${data.formatConfig.entityCount} items yourself.\n3. Search MULTIPLE times – first for the overall list, then individually for each item.\n\nDATA QUALITY:\n1. Never contradict user-provided facts.\n2. Always include exact numbers.\n3. Prioritize ESPN, official league sites, Billboard, IMDB, Wikipedia.\n4. CITATION INTEGRITY: Every URL in SOURCES must actually contain the cited fact.\n\nOUTPUT FORMAT FOR EACH ITEM:\n- Full name\n- 2-3 key stats with YEAR (from ${tc.lastSeason} season)\n- 1-2 achievements\n- One notable fact/quote\n- Source URL\n\nAt top: PRIMARY SOURCE URL\nAt bottom: SOURCES list`;
}

function buildPerplexityUser(data: AtomizedData): string {
  const tc = data.temporalContext;
  const mustBlock  = data.hasMustInclude ? `MANDATORY ITEMS:\n${data.mustIncludeItems.map((m, i) => `${i + 1}. ${m}`).join('\n')}\n\n` : '';
  const factsBlock = data.factOnlyRepresentation ? `EXISTING FACTS (Verify and ADD MORE. DO NOT CONTRADICT):\n${data.factOnlyRepresentation.substring(0, 1500)}\n\n` : '';

  return `DEEP RESEARCH for MSN slideshow.\n\nTitle: "${data.title}"\nCategory: ${data.category}\nItems needed: ${data.formatConfig.entityCount}\n\n${mustBlock}${factsBlock}SEARCH STRATEGY:\n1. Search: "${data.title} complete list"\n2. If no list found: autonomously identify ${data.formatConfig.entityCount} items.\n3. For each item: "[name] ${data.category} stats ${tc.lastSeason}"\n4. Search for additional hard numbers, exact dates, quotes.\n5. If any item has < 2 facts, search again.\n\nReturn EXACTLY ${data.formatConfig.entityCount} items with 3+ verified facts each.\n\nSTART WITH: PRIMARY SOURCE URL`;
}

export async function perplexityDeepResearch(data: AtomizedData): Promise<PerplexityRaw> {
  return callPerplexity('sonar', buildPerplexitySystem(data, true), buildPerplexityUser(data), 4000, 120_000);
}

export async function perplexityStandardResearch(data: AtomizedData): Promise<PerplexityRaw> {
  return callPerplexity('sonar', buildPerplexitySystem(data, false), buildPerplexityUser(data), 4000, 90_000);
}

export async function perplexityRetryResearch(data: AtomizedData): Promise<PerplexityRaw> {
  return callPerplexity('sonar-pro', buildPerplexitySystem(data, true), buildPerplexityUser(data), 4000, 120_000);
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

  let combinedFactRepresentation = '';
  if (data.factOnlyRepresentation) {
    combinedFactRepresentation = '=== FACTS FROM USER SOURCE (AUTHORITATIVE) ===\n' + data.factOnlyRepresentation + '\n\n';
  }
  combinedFactRepresentation += '=== FACTS FROM PERPLEXITY RESEARCH (SUPPLEMENTARY) ===\n' + answer.substring(0, 3000);
  if (data.userContext) {
    combinedFactRepresentation += '\n\n=== USER-PROVIDED CONTEXT (treat as direct source material) ===\n' + data.userContext;
  }

  return {
    ...data,
    perplexityAnswer:    answer,
    combinedFactRepresentation,
    primarySourceUrl,
    citations,
    sourceList: citations.map((url, i) => `[${i + 1}] ${url}`).join('\n'),
    researchWordCount: answer.split(/\s+/).length,
    researchOk: answer.length > 500 && citations.length >= 2,
  };
}

// ── 9. buildClaudePrompt (n8n: "Build Claude Prompt") ────────────────────────

export async function buildClaudePrompt(data: MergedData, citation1Markdown: string, citation2Markdown: string): Promise<PromptData> {
  function truncate(text: string, maxWords: number): string {
    const words = text.split(/\s+/);
    return words.length > maxWords ? words.slice(0, maxWords).join(' ') + '\n[truncated]' : text;
  }

  const citationContext = [
    citation1Markdown ? '\n\n=== SCRAPED CITATION SOURCE 1 ===\n' + truncate(citation1Markdown, 600) : '',
    citation2Markdown ? '\n\n=== SCRAPED CITATION SOURCE 2 ===\n' + truncate(citation2Markdown, 600) : '',
  ].filter(Boolean).join('');

  const BANNED_AI = `Delve, Embark, Foster, Navigate, Harness, Unlock, Elevate, Empower, Demystify, Catalyze, Optimize, Streamline, Tapestry, Landscape, Journey, Blueprint, Gateway, Intersection, Realm, Catalyst, Heartbeat, Pivotal, Comprehensive, Seamless, Vibrant, Dynamic, Synergistic, Multifaceted, Unparalleled, Robust, Transformative, Profound, Testament, Era, Synergy, "In today's world", "It is worth noting", "Moreover", "In conclusion", "Ultimately", "At the end of the day", "A testament to", "Furthermore", "Conversely", "Consequently", "In summary", "To wrap up", showcase, underscore, highlight, cement, solidify, storied, remarkable, notable, impressive, outstanding, exceptional, incredible, unparalleled, unprecedented, larger than life, household name, the rest is history`;

  const BANNED_CONTENT = `Nude, Naked, Suicide, Kill, Shot, Stabbed, Fake News, Misinformation, Conspiracy Theory, Hoax, Exploitation, Fetish, Adultery, Scandal, Trans, War, Terrorist, shit, Vaccination, Weed, Cannabis, Murder, Prison, Fraud, Conspiracy, Jail, Racist, Sex, Sexual, Mutilate, Pussy, Vagina, Dick, Penis, Sexy, Fuck, Harassment, Marijuana, Cocaine, Assault, Scam, Gambling, Drug, Racism, Allegation, Vaccine, Ganja, Battery, Laundering, Butt, ass, Betting, Pedophile, Rape, Molest, Damn, Faggot, Fag, Nigga, Bitch, Cigarette, Cigar, Cum, Dominatrix, Ejaculation, Genitals, Hooters, Jackass, Masturbate, Nipple, NSFW, Onlyfans, Opioids, Orgasm, Pedos, Piss, Porn, Schlong, Smoking, Spunk, Striptease, Testicle, Tobacco, Vibrator, WTF`;

  const ta = data.titleAnalysis;
  const fc = data.formatConfig;
  const tc = data.temporalContext;
  const writingStyleBlock = data.writingStyle ? `\n\nWRITING STYLE INFLUENCE (from writer): ${data.writingStyle}\nApply this style naturally throughout the slideshow.` : '';

  const claudeSystemPrompt = `You are an expert MSN Slideshow writer for American audiences covering ${data.category}.

${tc.dateAnchor}
${tc.seasonAnchor}${writingStyleBlock}

───────────────────────────────────────────────────────────────
ABSOLUTE NON-NEGOTIABLE OUTPUT RULE
───────────────────────────────────────────────────────────────

You MUST always produce the complete slideshow. No exceptions. Ever.

These responses are FORBIDDEN:
- "I need more data before I can proceed"
- "The fact database is insufficient"
- Flagging source problems instead of writing
- Any response that is not the full formatted slideshow

If source data is thin, use Tier 2 (training knowledge) and mark with [*].

───────────────────────────────────────────────────────────────
SOURCE HIERARCHY
───────────────────────────────────────────────────────────────

TIER 0 (LAW): User-provided source URL content. Overrides everything.
TIER 1: Perplexity research and scraped citations.
TIER 2: Well-known public facts from training. Mark with [*].
TIER 3 (NEVER): Speculation, invented stats, fabricated quotes.

───────────────────────────────────────────────────────────────
ORIGINALITY: Facts are yours to use. Language is NOT.
───────────────────────────────────────────────────────────────

───────────────────────────────────────────────────────────────
TITLE-BODY CORRELATION
───────────────────────────────────────────────────────────────

Title: "${data.title}"
- Numbers in title = exact count in body (${ta.promisedCount} items)
- Emotions (${ta.emotionalPromise ?? 'none detected'}) = explain WHO felt it, WHEN, WHY
- Main angle: ${ta.mainAngle}
${ta.secondaryAngle ? `- Secondary angle: ${ta.secondaryAngle}` : ''}

───────────────────────────────────────────────────────────────
WORD COUNTS (STRICT)
───────────────────────────────────────────────────────────────

- Meta Description: MAX 120 characters
- Intro slide (Slide 1): MAX 60 words
- Content slides: 35-50 words (aim for 40-45)

───────────────────────────────────────────────────────────────
META DESCRIPTION
───────────────────────────────────────────────────────────────

Max 120 characters. Intriguing. Angle-focused.
NEVER use: "Discover the...", "Explore the top...", "Find out why..."
Good pattern: [Specific unexpected fact]. [Implied question].

───────────────────────────────────────────────────────────────
INTRO SLIDE – MAX 60 WORDS
───────────────────────────────────────────────────────────────

Must: create curiosity, include ONE surprising fact, hint without naming items.
Must NOT: name list items, reveal #1, use "let's dive in" / "here are".

───────────────────────────────────────────────────────────────
WRITING VOICE
───────────────────────────────────────────────────────────────

Write like a sharp, witty sports columnist. React to facts; don't summarize them.
One-sentence gut punches. Contrast. Specificity over adjectives.
Alternate sentence lengths. Vary energy slide-to-slide.

Emotional textures to rotate: DISBELIEF, RESPECT, HUMOR (light), TENSION, NOSTALGIA.

Wikipedia voice = dead. Cheerleader voice = empty. Resume voice = stat dump.

───────────────────────────────────────────────────────────────
CONTENT SLIDES – 5Ws + 1H
───────────────────────────────────────────────────────────────

Every slide must answer relevant WHO/WHAT/WHEN/WHERE/WHY/HOW.
MAX 2 stats per slide. Every stat needs context (why it matters / when / who it affected).

───────────────────────────────────────────────────────────────
QUALITY CONSISTENCY ENGINE
───────────────────────────────────────────────────────────────

Before writing: assign an anchor fact to EVERY slide. If Tier 0/1 is thin, use Tier 2 and mark [*].
Quality tests per slide:
- Stranger Test: would a reader learn one specific real thing from this slide alone?
- Side-by-Side Test: is this as specific as slide 3?
- "So What?" Test: would a reader think "huh, I didn't know that"?

───────────────────────────────────────────────────────────────
VARIETY ENFORCEMENT
───────────────────────────────────────────────────────────────

Never start 2 consecutive slides with the same word.
Rotate sentence structures: [Stat+Context] / [Name+Action+Result] / [Time+What] / [Contrast+Fact]

${ta.isRanking ? `───────────────────────────────────────────────────────────────
RANKING ORDER – DESCENDING
───────────────────────────────────────────────────────────────

Slide 2 = rank ${ta.promisedCount} (lowest)
Last slide = rank 1 (best/top)
Each slide title starts with the rank number.
` : ''}${fc.isMultiSlideFormat ? `───────────────────────────────────────────────────────────────
MULTI-SLIDE FORMAT – 2 SLIDES PER ENTITY
───────────────────────────────────────────────────────────────

SLIDE A: WHO they are, PRIMARY achievement, key stat
SLIDE B: Supporting context, additional stats, legacy/impact. Must BUILD ON Slide A.
` : ''}
───────────────────────────────────────────────────────────────
PUNCTUATION BANS
───────────────────────────────────────────────────────────────

NO em-dashes (—) in any slide copy.
NO semicolons (;) in any slide copy.
NO ellipsis (...) in any slide copy.

───────────────────────────────────────────────────────────────
BANNED AI PHRASES
───────────────────────────────────────────────────────────────

${BANNED_AI}

───────────────────────────────────────────────────────────────
BANNED CONTENT WORDS
───────────────────────────────────────────────────────────────

${BANNED_CONTENT}
Profanity in direct quotes only: censor as first letter + asterisks.
Dick cannot be used anywhere, including in names.

───────────────────────────────────────────────────────────────
MSN SAFETY – 10-12 YEAR OLD TEST
───────────────────────────────────────────────────────────────

Before every slide ask: "Should a 10-12 year old be reading this?"

───────────────────────────────────────────────────────────────
FORMAT (Plain text only, no markdown)
───────────────────────────────────────────────────────────────

${data.title}

META: [Max 120 characters]

SLIDE 1
[Intro title]
[Max 60 words]

SLIDE 2
[Creative title${ta.isRanking ? ' – start with rank number' : ''}]
[35-50 words]

...continue for all ${data.slideCount} slides...

SOURCES:
[URL]: [What facts came from this source]`;

  const claudeUserPrompt = `FACT DATABASE – Use these facts, write your OWN words:

${data.combinedFactRepresentation}${citationContext}

───────────────────────────────────────────────────────────────
ASSIGNMENT
───────────────────────────────────────────────────────────────

Title: "${data.title}"
Category: ${data.category}
Slides: 1 intro + ${data.slideCount} content slides
${fc.isMultiSlideFormat ? `Format: ${fc.slidesPerEntity} slides per entity (${fc.entityCount} entities total)` : ''}
${data.hasMustInclude ? `\nMANDATORY ITEMS (must all appear):\n${data.mustIncludeItems.map((m, i) => `${i + 1}. ${m}`).join('\n')}` : ''}

Primary Source: ${data.primarySourceUrl || 'Use Perplexity research'}

PRE-WRITE CHECKLIST:
1. What exact promise does the title make?
2. Main angle vs secondary angle?
3. For EACH slide, what is the single strongest anchor fact?
4. How will I vary structure and opening of each slide?
${ta.isRanking ? '5. Am I ordering slides from lowest rank to highest?' : ''}

REMINDER: Output the complete slideshow no matter what. Mark Tier 2 facts with [*].

Write the complete slideshow now.
- NO em-dashes, semicolons, or ellipsis anywhere
- Include SOURCES section at the end`;

  return { ...data, claudeSystemPrompt, claudeUserPrompt };
}

// ── 10. generateWithClaude (n8n: "Claude - Generate Article") ─────────────────

export async function generateWithClaude(systemPrompt: string, userPrompt: string): Promise<unknown> {
  try {
    const resp = await axios.post(
      'https://api.anthropic.com/v1/messages',
      {
        model: 'claude-sonnet-4-20250514',
        max_tokens: 6000,
        temperature: 0.3,
        system: [{ type: 'text', text: systemPrompt, cache_control: { type: 'ephemeral' } }],
        messages: [{ role: 'user', content: userPrompt }],
      },
      {
        headers: {
          'Content-Type': 'application/json',
          'anthropic-beta': 'prompt-caching-2024-07-31',
          'anthropic-version': '2023-06-01',
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
    { model: 'grok-4.20-0309-reasoning', max_tokens: 6000, temperature: 0.3, messages: [{ role: 'system', content: systemPrompt }, { role: 'user', content: userPrompt }] },
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
  if (metaText.length > 120) warnings.push(`Meta too long: ${metaText.length} chars (max 120)`);
  if (/find out|discover|see which|here are|we rank|you won't believe|explore/i.test(metaText)) warnings.push('Meta uses AI-generated patterns');

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

  const validationStatus = errors.length > 0 ? 'FAILED' : (warnings.length > 0 ? 'WARNINGS' : 'PASSED');

  return {
    ...data,
    articleText,
    structuralValidation: { status: validationStatus, errors, warnings, autoFixes, slideCount: slides.length, metaLength: metaText.length },
    plagiarismCheck:      { score: plagScore, matches: plagMatches.slice(0, 10), status: plagScore > 30 ? 'HIGH' : plagScore > 10 ? 'MEDIUM' : 'LOW' },
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

// ── 15. perplexityVerifyClaims (n8n: "Perplexity - Verify Claims") ────────────

export async function perplexityVerifyClaims(data: ClaimedData): Promise<PerplexityRaw> {
  const systemContent = `You are a fact-checker with LIVE web search.\n\n${data.temporalContext.dateAnchor}\n\nFor EACH claim respond:\nCLAIM [number]: [the claim]\nSTATUS: [VERIFIED / INCORRECT / UNVERIFIABLE]\nFINDING: [What you found]\nSOURCE: [URL]`;
  const userContent   = `Verify these claims from an MSN article:\n\n${data.claimsToVerify.map((c, i) => `${i + 1}. [${c.type}] "${c.claim}" – Context: "${c.context}"`).join('\n')}\n\nSearch and verify each one. Provide source URLs.`;
  return callPerplexity('sonar', systemContent, userContent, 3000, 90_000);
}

// ── 16. processVerification (n8n: "Process Verification") ────────────────────

export async function processVerification(data: ClaimedData, verifyResp: PerplexityRaw): Promise<VerifiedData> {
  const text      = verifyResp?.choices?.[0]?.message?.content ?? '';
  const citations = verifyResp?.citations ?? [];
  const results: VerifiedData['perplexityVerification']['results'] = [];

  const pattern = /CLAIM\s*(\d+):\s*([^\n]+)\nSTATUS:\s*(VERIFIED|INCORRECT|UNVERIFIABLE)\nFINDING:\s*([^\n]+)\nSOURCE:\s*([^\n]+)/gi;
  let m: RegExpExecArray | null;
  while ((m = pattern.exec(text)) !== null) {
    results.push({ claimIndex: parseInt(m[1]) - 1, claim: m[2].trim(), status: m[3].toUpperCase(), finding: m[4].trim(), source: m[5].trim() });
  }

  const verified    = results.filter(r => r.status === 'VERIFIED').length;
  const incorrect   = results.filter(r => r.status === 'INCORRECT').length;
  const unverifiable = results.filter(r => r.status === 'UNVERIFIABLE').length;

  return {
    ...data,
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
  const systemContent = [
    {
      type: 'text',
      text: `You are a STRICT MSN Slideshow Auditor with LIVE web search access.\n\nJOB 1: INDEPENDENT FACT VERIFICATION\nFor EVERY specific claim (stats, dates, records, achievements), search the web independently and verify against authoritative sources.\n\nJOB 2: RULE COMPLIANCE AUDIT\nSTRUCTURE: Intro max 60 words? Content slides 35-50 words? Meta max 120 chars?\nTITLE-BODY: Every promise in title delivered?\nINTRO: Has specific fact? Has hook without revealing list items?\nPUNCTUATION: No em-dashes, semicolons, or ellipsis in slide copy?\nBANNED PHRASES: Check for banned words and remove them.\nMSN SAFETY: Safe for 10-12 year old?\n\nOUTPUT FORMAT:\n\nCORRECTED ARTICLE\n[Full article with fixes applied]\n\n=== FACT VERIFICATION (GROK) ===\n[Per-slide verification]\n\n=== MASTER SOURCE LIST ===\n[Numbered URLs with facts verified]\n\n=== RULE COMPLIANCE ===\n[PASS/FAIL per rule]\n\n=== AUDIT SUMMARY ===\nFacts verified by Grok: X/Y\nRule violations found: Z\nAuto-corrections applied: [list]\nFlags for human review: [list]\n=== END AUDIT ===`,
      cache_control: { type: 'ephemeral' },
    },
    {
      type: 'text',
      text: `${tc.dateAnchor}\n${tc.seasonAnchor}\n\nPay special attention to:\n- Stats that seem rounded\n- Dates/years (last season = ${tc.lastSeason})\n- Rankings and superlatives\n- Quotes – verify exact wording`,
    },
  ];

  const resp = await axios.post(
    'https://api.x.ai/v1/chat/completions',
    {
      model: 'grok-3-latest', max_tokens: 6500, temperature: 0.0,
      messages: [
        { role: 'system', content: systemContent },
        { role: 'user',   content: `AUDIT THIS MSN SLIDESHOW:\n\nTitle: "${data.title}"\nCategory: ${data.category}\nExpected slides: ${data.slideCount}\nPrimary Source: ${data.primarySourceUrl}\n\nARTICLE TO AUDIT:\n${data.articleText}\n\nPerform independent fact verification AND rule compliance audit. Output in exact format specified.` },
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
      grokAudit: { status: 'FAILED', factVerification: '', ruleCompliance: '', summary: '', stats: { factsVerified: 'N/A', violations: 0, corrections: 'None', flags: 'None' } },
      grokSources: [], combinedSourceList: [], combinedSourceListText: '',
      rewriteApplied: false,
    };
  }

  const articleMatch     = grokText.match(/CORRECTED ARTICLE\s*([\s\S]*?)(?====\s*FACT VERIFICATION|$)/i);
  const factVerifyMatch  = grokText.match(/=== FACT VERIFICATION \(GROK\) ===([\s\S]*?)(?====\s*MASTER SOURCE LIST|$)/i);
  const sourceListMatch  = grokText.match(/=== MASTER SOURCE LIST ===([\s\S]*?)(?====\s*RULE COMPLIANCE|$)/i);
  const complianceMatch  = grokText.match(/=== RULE COMPLIANCE ===([\s\S]*?)(?====\s*AUDIT SUMMARY|$)/i);
  const summaryMatch     = grokText.match(/=== AUDIT SUMMARY ===([\s\S]*?)(?====\s*END AUDIT|$)/i);

  const auditedArticle   = articleMatch?.[1]?.trim() ?? '';
  const factVerification = factVerifyMatch?.[1]?.trim() ?? '';
  const ruleCompliance   = complianceMatch?.[1]?.trim() ?? '';
  const auditSummary     = summaryMatch?.[1]?.trim() ?? '';
  const masterSourceList = sourceListMatch?.[1]?.trim() ?? '';

  // Parse Grok sources
  const grokSources: SourceEntry[] = [];
  const srcPattern = /(\d+)\.\s*(https?:\/\/[^\s]+)\s*[-–]\s*Facts verified:\s*([^\n]+)/gi;
  let srcMatch: RegExpExecArray | null;
  while ((srcMatch = srcPattern.exec(masterSourceList)) !== null) {
    grokSources.push({ index: parseInt(srcMatch[1]), url: srcMatch[2].trim(), factsVerified: srcMatch[3].trim(), verifiedBy: 'Grok' });
  }

  // Merge with Perplexity sources
  const perplexitySources: SourceEntry[] = (data.perplexityVerification?.citations ?? []).map((url, i) => ({
    index: i + 1, url, verifiedBy: 'Perplexity',
    factsVerified: (data.perplexityVerification?.results?.filter(r => r.source === url).map(r => r.claim).join(', ')) || 'General verification',
  }));

  const allSources = [...perplexitySources];
  grokSources.forEach(gs => {
    const existing = allSources.find(ps => ps.url === gs.url);
    if (existing) { existing.factsVerified += ` + Grok: ${gs.factsVerified}`; existing.verifiedBy = 'Both'; }
    else allSources.push(gs);
  });

  const combinedSourceListText = allSources.map((s, i) => `${i + 1}. ${s.url}\n   Verified by: ${s.verifiedBy}\n   Facts: ${s.factsVerified}`).join('\n\n');

  const factsVerifiedMatch = auditSummary.match(/Facts verified by Grok:\s*(\d+)\/(\d+)/i);
  const violationsMatch    = auditSummary.match(/Rule violations found:\s*(\d+)/i);
  const correctionsMatch   = auditSummary.match(/Auto-corrections applied:\s*([^\n]+)/i);
  const flagsMatch         = auditSummary.match(/Flags for human review:\s*([^\n]+)/i);

  const finalArticle = auditedArticle.length > 500 ? auditedArticle : originalArticle;

  return {
    ...data,
    originalArticleText: originalArticle,
    articleText: finalArticle,
    grokAudit: {
      status: 'COMPLETED', factVerification, ruleCompliance, summary: auditSummary,
      stats: {
        factsVerified: factsVerifiedMatch ? `${factsVerifiedMatch[1]}/${factsVerifiedMatch[2]}` : 'N/A',
        violations:    violationsMatch ? parseInt(violationsMatch[1]) : 0,
        corrections:   correctionsMatch?.[1]?.trim() ?? 'None',
        flags:         flagsMatch?.[1]?.trim() ?? 'None',
      },
    },
    grokSources, combinedSourceList: allSources, combinedSourceListText,
    rewriteApplied: finalArticle !== originalArticle,
  };
}

// ── 19. finalAssembly (n8n: "Final Assembly") ────────────────────────────────

export async function finalAssembly(data: AuditedData): Promise<FinalOutput> {
  const researchScore    = data.researchOk ? 80 : 50;
  const perplexityScore  = data.perplexityVerification?.score ?? 50;
  const structuralScore  = data.structuralValidation?.status === 'PASSED' ? 100 : data.structuralValidation?.status === 'WARNINGS' ? 70 : 40;
  const plagiarismScore  = data.plagiarismCheck?.status === 'LOW' ? 100 : data.plagiarismCheck?.status === 'MEDIUM' ? 70 : 40;
  const qualityScore     = Math.round((researchScore * 0.2) + (perplexityScore * 0.3) + (structuralScore * 0.25) + (plagiarismScore * 0.25));

  const summaryParts: string[] = [];
  if (data.structuralValidation?.errors?.length)  summaryParts.push(`Errors: ${data.structuralValidation.errors.join('; ')}`);
  if (data.structuralValidation?.warnings?.length) summaryParts.push(`Warnings: ${data.structuralValidation.warnings.length}`);
  if (data.rewriteApplied)                         summaryParts.push('Grok corrections applied');
  if (data.grokAudit?.stats?.flags && data.grokAudit.stats.flags !== 'None') summaryParts.push(`Flags: ${data.grokAudit.stats.flags}`);
  if (data.generatedBy)                            summaryParts.push(`Generated by: ${data.generatedBy}`);

  const auditReport = `
───────────────────────────────────────────────────────────────
QUALITY AUDIT REPORT
───────────────────────────────────────────────────────────────

OVERALL QUALITY SCORE: ${qualityScore}/100

RESEARCH:        ${researchScore}/100
VERIFICATION:    ${perplexityScore}/100
STRUCTURE:       ${structuralScore}/100
ORIGINALITY:     ${plagiarismScore}/100

PERPLEXITY VERIFICATION
Verified:     ${data.perplexityVerification?.stats?.verified ?? 0}
Incorrect:    ${data.perplexityVerification?.stats?.incorrect ?? 0}
Unverifiable: ${data.perplexityVerification?.stats?.unverifiable ?? 0}

GROK AUDIT
Facts Verified: ${data.grokAudit?.stats?.factsVerified ?? 'N/A'}
Violations:     ${data.grokAudit?.stats?.violations ?? 0}
Corrections:    ${data.grokAudit?.stats?.corrections ?? 'None'}
Flags:          ${data.grokAudit?.stats?.flags ?? 'None'}

STRUCTURAL VALIDATION
Status:    ${data.structuralValidation?.status ?? 'N/A'}
Errors:    ${data.structuralValidation?.errors?.join(', ') || 'None'}
Warnings:  ${data.structuralValidation?.warnings?.join(', ') || 'None'}
Auto-fixes: ${data.structuralValidation?.autoFixes?.join(', ') || 'None'}

VERIFICATION SOURCES
${data.combinedSourceListText || 'No sources collected'}

───────────────────────────────────────────────────────────────
`;

  return {
    title: data.title, category: data.category, slideCount: data.slideCount, writerName: data.writerName,
    articleText:         data.articleText,
    originalArticleText: data.originalArticleText || data.articleText,
    auditReport, qualityScore, researchScore,
    verificationScore: perplexityScore, structuralScore, originalityScore: plagiarismScore,
    primarySourceUrl:       data.primarySourceUrl,
    combinedSourceListText: data.combinedSourceListText,
    summaryComment:  summaryParts.length > 0 ? summaryParts.join('. ') : 'Article passed all checks.',
    validationStatus: data.structuralValidation?.status ?? 'UNKNOWN',
    factsVerified:     `${data.perplexityVerification?.stats?.verified ?? 0}/${data.perplexityVerification?.stats?.total ?? 0}`,
    grokFactsVerified: data.grokAudit?.stats?.factsVerified ?? 'N/A',
    flagsForReview:    data.grokAudit?.stats?.flags ?? 'None',
    generatedBy:  data.generatedBy ?? 'Unknown',
    generatedAt:  new Date().toISOString(),
  };
}

