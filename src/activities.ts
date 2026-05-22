import axios from 'axios';
import * as dotenv from 'dotenv';
import {
  FormInput, PreparedData, SourcedData, AtomizedData, ResearchedData,
  MergedData, PromptData, GeneratedData, ValidatedData, ClaimedData,
  VerifiedData, AuditedData, FinalOutput, TemporalCtx,
  FormatConfig, TitleAnalysis, AtomizedFact, SourceEntry, FactProvenance,
  SubjectivePreparedData, SubjectiveResearchedData, SubjectiveMergedData,
  SubjectivePromptData, SubjectiveGeneratedData, SubjectiveValidatedData,
  SubjectiveAuditedData,
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

// ─────────────────────────────────────────────────────────────────────────────
// Surgical audit patches — shared by both subjective and objective Grok audits.
// Grok emits find/replace patches; we apply them to the original article so
// the slide count and structure can never be corrupted by the audit step.
// ─────────────────────────────────────────────────────────────────────────────

export interface AuditPatch {
  scope: string;          // "SLIDE 5" | "META" | "INTRO" | "TITLE"
  find: string;           // verbatim text to locate in the original article
  replace: string;        // replacement text
  reason: string;         // brief explanation for logs
}

// Block-delimited format that the audit prompts ask Grok to emit:
//   <<<PATCH>>>
//   SCOPE: SLIDE 5
//   FIND:
//   <original verbatim text>
//   END_FIND
//   REPLACE:
//   <fixed text>
//   END_REPLACE
//   REASON: brief
//   <<<END>>>
const PATCH_BLOCK_RE = /<<<PATCH>>>([\s\S]*?)<<<END>>>/gi;
const PATCH_SCOPE_RE   = /SCOPE:\s*([^\n]+)/i;
const PATCH_FIND_RE    = /FIND:\s*\n([\s\S]*?)\nEND_FIND/i;
const PATCH_REPLACE_RE = /REPLACE:\s*\n?([\s\S]*?)\nEND_REPLACE/i;
const PATCH_REASON_RE  = /REASON:\s*([^\n]+)/i;

export function parseAuditPatches(auditText: string): AuditPatch[] {
  const patches: AuditPatch[] = [];
  let m: RegExpExecArray | null;
  PATCH_BLOCK_RE.lastIndex = 0;
  while ((m = PATCH_BLOCK_RE.exec(auditText)) !== null) {
    const block = m[1];
    const scope   = block.match(PATCH_SCOPE_RE)?.[1]?.trim();
    const find    = block.match(PATCH_FIND_RE)?.[1];
    const replace = block.match(PATCH_REPLACE_RE)?.[1];
    if (!scope || find == null || replace == null) continue;
    patches.push({
      scope,
      find:    find.replace(/\r/g, ''),
      replace: replace.replace(/\r/g, ''),
      reason:  block.match(PATCH_REASON_RE)?.[1]?.trim() ?? '',
    });
  }
  return patches;
}

export function applyAuditPatches(article: string, patches: AuditPatch[]): { result: string; applied: number; skipped: number; log: string[] } {
  let result = article;
  let applied = 0;
  let skipped = 0;
  const log: string[] = [];

  for (const p of patches) {
    if (!p.find.trim()) { skipped++; log.push(`SKIP [${p.scope}] empty FIND`); continue; }
    // Try exact match first, then a whitespace-normalized fallback.
    const before = result;
    if (result.includes(p.find)) {
      result = result.replace(p.find, p.replace);
    } else {
      // Collapse multiple whitespace in both haystack and needle so trivial
      // formatting differences don't make the patch miss.
      const norm = (s: string) => s.replace(/\s+/g, ' ').trim();
      const needle = norm(p.find);
      const haystack = norm(result);
      const idx = haystack.indexOf(needle);
      if (idx >= 0) {
        // Find the matching span in the original (un-normalized) text by
        // walking word-by-word. Simpler: do a regex with \s+ between every word.
        const escaped = p.find.split(/\s+/).map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('\\s+');
        const rx = new RegExp(escaped);
        if (rx.test(result)) result = result.replace(rx, p.replace);
      }
    }
    if (result === before) {
      skipped++;
      log.push(`SKIP [${p.scope}] FIND not located: ${p.find.slice(0, 80).replace(/\n/g, ' ')}`);
    } else {
      applied++;
      log.push(`APPLY [${p.scope}] ${p.reason || '(no reason)'}`);
    }
  }
  return { result, applied, skipped, log };
}

// ── 1. prepareInputAndAnalyze (n8n: "Prepare Input & Analyze") ────────────────

export async function prepareInputAndAnalyze(input: FormInput): Promise<PreparedData> {
  const {
    writerName, title: rawTitle, category: rawCategory, slideCount: rawSlides,
    slidesPerEntityRaw, sourcesRaw, mustIncludeRaw, userContext, writingStyle,
  } = input;

  const title    = rawTitle.trim();
  const category = rawCategory.trim();
  const slideCount = typeof rawSlides === 'string' ? parseInt(rawSlides, 10) || 20 : (rawSlides || 20);

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

  // Source parsing — accept up to 5 URLs, handle commas, newlines, OR spaces
  // within a line (n8n's flatMap pattern). Drop restricted domains here so the
  // workflow only iterates URLs that are actually scrape-worthy.
  const allUrls = sourcesRaw
    .split(/[\n\r,]+/)
    .flatMap(chunk => chunk.trim().split(/\s+/))
    .map(s => s.trim())
    .filter(s => /^https?:\/\/.+\..+/.test(s))
    .filter(s => !isRestricted(s))
    .slice(0, 5);

  const userPrimaryUrl      = allUrls[0] ?? '';
  const userSecondaryUrls   = allUrls.slice(1);                // up to 4 extras (5 total)
  const isUserUrlRestricted = userPrimaryUrl ? isRestricted(userPrimaryUrl) : false;
  const hasValidUserSource  = !!userPrimaryUrl && !isUserUrlRestricted;
  const mustIncludeItems    = mustIncludeRaw ? mustIncludeRaw.split(/[\n,]+/).map(s => s.trim()).filter(Boolean) : [];

  return {
    title, category, slideCount, writerName, userContext, writingStyle,
    userPrimaryUrl, userSecondaryUrls, hasValidUserSource, isUserUrlRestricted,
    restrictedDomains: RESTRICTED_DOMAINS, mustIncludeItems,
    hasMustInclude: mustIncludeItems.length > 0,
    temporalContext, formatConfig, titleAnalysis,
    sourceCount: allUrls.length,
    timestamp: new Date().toISOString(),
    isSports: category.startsWith('Sports'),
  };
}

// ── 2. firecrawlScrape (n8n: Firecrawl nodes) ────────────────────────────────
//
// Uses Firecrawl v2 with markdown output (matches n8n) plus three tuning knobs
// n8n doesn't pass:
//   - waitFor: 2000        → JS-heavy sites (givemesport, sportskeeda) render
//                             their article DOM after a delay; without waiting
//                             Firecrawl reads only the initial video-player
//                             skeleton
//   - blockAds: true       → strips ad-iframe content before markdown convert
//   - excludeTags expanded → also strips video/audio/iframe/figure/dialog/form
//                             which on modern sites produce nav-menu garbage
//
// The result always passes through cleanFirecrawlMarkdown() — see comment
// below. Content is ALWAYS returned; an unusable scrape would still be passed
// downstream where the alignment scorer + atomizer + Perplexity fallback
// already handle thin sources gracefully.

const FIRECRAWL_EXCLUDE_TAGS = [
  // n8n's original set
  'nav', 'footer', 'aside', 'header', 'script', 'style', 'ads', 'comments',
  // Embedded media that produces player-UI noise (givemesport, sportskeeda…)
  'video', 'audio', 'iframe', 'source', 'track', 'picture', 'svg', 'canvas',
  // Interactive chrome
  'form', 'button', 'input', 'select', 'option', 'dialog', 'noscript',
];

// ─────────────────────────────────────────────────────────────────────────────
// cleanFirecrawlMarkdown — dedicated parser/cleaner step.
//
// This is the equivalent of an n8n "Code node" that processes the Firecrawl
// output before it feeds the AI prompts. It is NON-DESTRUCTIVE to article
// prose — only well-known UI noise patterns are removed.
//
// Patterns stripped:
//   - Video-player chrome (Now Playing, Play, Mute, Fullscreen, Loaded: 0%,
//     Stream Type LIVE, Picture-in-Picture, This is a modal window, etc.)
//   - Ad widget labels (✕ Remove Ads, Skip Ad, Advertisement, WATCH NEXT)
//   - Cookie/consent banners (We use cookies, Accept All, Manage Settings)
//   - Newsletter signup widgets (Subscribe, Sign up for our newsletter)
//   - Image-only lines with no alt text ([](url) → drops)
//   - Excessive blank lines collapsed to one
//
// Any line with substantive prose (>40 chars, sentence-shaped) is always kept.
// ─────────────────────────────────────────────────────────────────────────────

const NOISE_LINE_PATTERNS: RegExp[] = [
  // Video player UI
  /^Close\s*$/i,
  /^(?:WATCH|NEXT|Now Playing|Play Video|Play|Pause|Unmute|Mute|Fullscreen|Skip Ad|Advertisement|Visit)\s*$/i,
  /^Video Player is loading\.?\s*$/i,
  /^Current Time\s+[\d:\-\.\s\/]+$/i,
  /^Duration\s+[\d:\-\.\s\/]+$/i,
  /^Loaded:\s*\d+%\s*$/i,
  /^Stream Type\s+\S+\s*$/i,
  /^Seek to live.*$/i,
  /^Remaining Time.*$/i,
  /^\d+(?:\.\d+)?x\s*$/,
  /^Playback Rate\s*$/i,
  /^-?\s*(?:Chapters|Descriptions|Captions|Audio Track)\s*$/i,
  /^-\s*(?:descriptions|captions)\s+off,?\s*selected\s*$/i,
  /^Picture-in-Picture.*$/i,
  /^This is a modal window\.?\s*$/i,
  // Ad/promo widgets
  /^[✕✖×]\s*(?:Remove Ads)?.*$/,
  /^UELFACMLSEPLUCL\s*$/i,
  /^(?:Sponsored|Promoted|Recommended for you|Related Articles?)\s*$/i,
  // Cookie banners
  /^(?:Accept|Accept All|Reject|Reject All|Manage Settings|Cookie Settings|We use cookies)\b.*$/i,
  // Newsletter signups
  /^(?:Subscribe|Sign up|Get the latest|Join (?:our )?newsletter|Enter your email)\b.*$/i,
  // Social-share buttons
  /^(?:Share|Tweet|Share on (?:Facebook|Twitter|LinkedIn|Reddit))\s*$/i,
  // Image-only markdown lines: ![alt](url) or [![](url)](link) with no text
  /^\[?!\[\]\([^)]*\)\]?\([^)]*\)?\s*$/,
];

export function cleanFirecrawlMarkdown(md: string): string {
  if (!md) return '';
  return md
    .split('\n')
    .filter(line => {
      const t = line.trim();
      if (!t) return true;                                  // keep blank lines (paragraph breaks)
      if (t.length > 40) return true;                       // prose always passes
      return !NOISE_LINE_PATTERNS.some(rx => rx.test(t));
    })
    .join('\n')
    .replace(/\n{3,}/g, '\n\n')                             // collapse 3+ blank lines → 1
    .trim();
}

export async function firecrawlScrape(url: string, onlyMainContent = true): Promise<string> {
  if (!url || !url.startsWith('http')) {
    console.log(`[firecrawlScrape] Skipping URL (empty/invalid): ${url}`);
    return '';
  }
  if (!FIRECRAWL_KEY) {
    console.error('[firecrawlScrape] FIRECRAWL_API_KEY is not set — skipping scrape');
    return '';
  }

  const payload = {
    url,
    formats: ['markdown'] as const,
    onlyMainContent,
    excludeTags: FIRECRAWL_EXCLUDE_TAGS,
    blockAds: true,
    removeBase64Images: true,
    waitFor: 2000,                                       // let JS render before reading DOM
  };
  const headers = { Authorization: `Bearer ${FIRECRAWL_KEY}` };

  // Attempt up to 2 tries — retry once on 429 rate-limit or 5xx server errors
  for (let attempt = 1; attempt <= 2; attempt++) {
    try {
      const resp = await axios.post(
        'https://api.firecrawl.dev/v2/scrape',
        payload,
        { headers, timeout: 75_000 },
      );

      const raw = resp.data?.data?.markdown ?? resp.data?.markdown ?? '';
      const cleaned = cleanFirecrawlMarkdown(raw);

      if (!cleaned) {
        console.warn(`[firecrawlScrape] Empty markdown for ${url} (status ${resp.status}, success=${resp.data?.success}, attempt ${attempt})`);
      } else {
        console.log(`[firecrawlScrape] ${url}: ${raw.length} chars raw → ${cleaned.length} chars cleaned (attempt ${attempt})`);
      }
      return cleaned;
    } catch (err: unknown) {
      const status = axios.isAxiosError(err) ? err.response?.status : undefined;
      const errBody = axios.isAxiosError(err) ? JSON.stringify(err.response?.data ?? err.message) : String(err);

      if (status === 402) {
        console.error(`[firecrawlScrape] ⚠ FIRECRAWL CREDITS EXHAUSTED (402) for ${url}. Check your Firecrawl billing dashboard.`);
        return '';                                       // no point retrying
      }
      if (status === 401) {
        console.error(`[firecrawlScrape] ⚠ FIRECRAWL API KEY INVALID (401) for ${url}. Check FIRECRAWL_API_KEY env var.`);
        return '';                                       // no point retrying
      }

      console.error(`[firecrawlScrape] HTTP ${status ?? '?'} for ${url} (attempt ${attempt}): ${errBody}`);

      // Retry on 429 (rate limit) or 5xx (server error), but not on other errors
      if (attempt < 2 && (status === 429 || (status && status >= 500))) {
        const delay = status === 429 ? 5000 : 3000;
        console.log(`[firecrawlScrape] Retrying ${url} in ${delay}ms...`);
        await new Promise(r => setTimeout(r, delay));
        continue;
      }
      return '';
    }
  }
  return '';
}

// ── 3. analyzeSourceAlignment (n8n: "Analyze Source Alignment") ──────────────

export async function analyzeSourceAlignment(
  prepData: PreparedData,
  primaryMarkdown: string,
  additionalMarkdowns: string[] = [],
): Promise<SourcedData> {
  // Build content from primary + every additional source (up to 4 extras).
  const sourceSections: string[] = [];
  if (primaryMarkdown) sourceSections.push(`=== PRIMARY SOURCE: ${prepData.userPrimaryUrl} ===\n${primaryMarkdown}`);
  additionalMarkdowns.forEach((md, i) => {
    if (md && md.length > 0) {
      const url = prepData.userSecondaryUrls[i] ?? '';
      const label = i === 0 ? 'SECONDARY' : i === 1 ? 'TERTIARY' : `SOURCE ${i + 2}`;
      sourceSections.push(`=== ${label} SOURCE: ${url} ===\n${md}`);
    }
  });
  const allSourceContent = sourceSections.join('\n\n');

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
      secondaryLength: additionalMarkdowns.reduce((s, m) => s + m.length, 0),
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

  // Try multiple section-splitting strategies in priority order
  let sections: string[] = [];
  if (sourceContent) {
    // Strategy 1: "N of M" slideshow format (Yardbarker, Bleacher Report, etc.)
    // Splits on lines like "1 of 32", "2 of 32" that precede ## headings
    const nOfMSections = sourceContent.split(/(?=\n\d{1,3}\s+of\s+\d{1,3}\s*\n)/);
    if (nOfMSections.length >= 3) {
      sections = nOfMSections;
    }

    // Strategy 2: ## headers with linked or plain "Entity: Subject" format
    // e.g. "## [Arizona Cardinals](url): Karlos Dansby" or "## Arizona Cardinals: Karlos Dansby"
    if (sections.length < 3) {
      const entitySections = sourceContent.split(/(?=##\s*\[?[A-Z])/);
      if (entitySections.length >= 3) {
        sections = entitySections;
      }
    }

    // Strategy 3: Original numbered format (## 1. Title, # 2: Title)
    if (sections.length < 3) {
      sections = sourceContent.split(/(?=##?\s*\d+[.):]?\s*)/);
    }
  }

  sections.forEach((section, sectionIdx) => {
    if (section.trim().length < 30) return;

    // Extract item number and name with multiple patterns
    let itemNumber = sectionIdx + 1;
    let itemName   = `Item ${sectionIdx + 1}`;
    let content    = section;

    // Pattern A: "N of M" counter line followed by ## heading
    const nOfMMatch = section.match(/(\d{1,3})\s+of\s+\d{1,3}\s*\n+##\s*\[?([^\]\n]+)\]?(?:\([^)]*\))?[:\s]*(.+?)(?:\n|$)/);
    if (nOfMMatch) {
      itemNumber = parseInt(nOfMMatch[1]);
      const team = nOfMMatch[2].trim();
      const subject = nOfMMatch[3]?.trim().replace(/\*\*/g, '').replace(/\[([^\]]+)\]\([^)]*\)/g, '$1') ?? '';
      itemName = subject ? `${team}: ${subject}` : team;
      content = section.replace(/^[\s\S]*?\n##\s*[^\n]+\n/, '');
    } else {
      // Pattern B: ## [Team](url): Player or ## Team: Player
      const headingMatch = section.match(/^##\s*\[?([^\]\n]+)\]?(?:\([^)]*\))?[:\s]+\[?([^\]\n(]+)\]?(?:\([^)]*\))?(?:\n|$)/);
      if (headingMatch) {
        const team = headingMatch[1].trim();
        const subject = headingMatch[2]?.trim().replace(/\*\*/g, '').replace(/\[([^\]]+)\]\([^)]*\)/g, '$1') ?? '';
        itemName = subject ? `${team}: ${subject}` : team;
        content = section.replace(/^##\s*[^\n]+\n/, '');
      } else {
        // Pattern C: Original numbered format (## 1. Title)
        const numberedMatch = section.match(/^##?\s*(\d+)[.):]?\s*(.+?)(?:\n|$)/);
        if (numberedMatch) {
          itemNumber = parseInt(numberedMatch[1]);
          itemName = numberedMatch[2].trim().replace(/\*\*/g, '');
          content = section.replace(/^##?\s*\d+[.):]?\s*.+?\n/, '');
        }
      }
    }

    // Strip markdown images, link markup, and image credit lines from content
    content = content
      .replace(/!\[[^\]]*\]\([^)]*\)/g, '')
      .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
      .replace(/^[A-Z][\w. ]+(?:\/|-)?(?:Getty Images|Icon Sportswire|Imagn Images|USA TODAY Sports|Allsport|Getty|AP Photo|Icon SMI)[^\n]*\s*/gm, '')
      .trim();
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
  const softRefusals = ['has not yet occurred','not yet been announced','fragmented and incomplete','corrupted or partially-rendered'];
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

  // Detect if atomization failed despite having a rich source
  const atomizationItemCount = data.atomizedFacts?.filter(f => f.itemName !== 'USER_CONTEXT_DATA').length ?? 0;
  const atomizationFailed = userSourceContent.length > 2000 && atomizationItemCount <= 1;
  if (atomizationFailed) {
    console.warn(`[mergeResearch] Atomization produced only ${atomizationItemCount} item(s) from ${userSourceContent.length} chars of source — full source text will be used directly`);
  }

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
    if (data.factOnlyRepresentation && !atomizationFailed) {
      combinedFactRepresentation += '### Atomized Facts\n' + data.factOnlyRepresentation + '\n\n';
    } else if (atomizationFailed) {
      combinedFactRepresentation += '### NOTE: Atomized facts were not available. The full source text below IS your Tier 1A fact database. Read it carefully to extract all items, stats, and facts.\n\n';
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

═══════════════════════════════════════════════════════════════
BANNED AI PHRASES — NEVER USE
═══════════════════════════════════════════════════════════════

${BANNED_AI}

═══════════════════════════════════════════════════════════════
BANNED CONTENT WORDS — NEVER USE
═══════════════════════════════════════════════════════════════

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
Slides: 1 intro + ${data.slideCount} content slides (MANDATORY — you MUST produce exactly ${data.slideCount} content slides, no more, no fewer. If the source material covers fewer than ${data.slideCount} items, add additional slides with honorable mentions, historical context, or related entries to reach exactly ${data.slideCount}.)
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
  const model = 'claude-sonnet-4-5-20250929';
  console.log(`[generateWithClaude] Calling model: ${model}`);
  try {
    const resp = await axios.post(
      'https://api.anthropic.com/v1/messages',
      {
        model,
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
    console.log(`[generateWithClaude] Success — response type: ${resp.data?.type}, stop_reason: ${resp.data?.stop_reason}`);
    return resp.data;
  } catch (err: unknown) {
    if (axios.isAxiosError(err)) {
      console.error(`[generateWithClaude] HTTP ${err.response?.status}: ${JSON.stringify(err.response?.data ?? err.message)}`);
      return err.response?.data ?? { type: 'error', error: { message: String(err.message) } };
    }
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
  const bannedPhrases = ['delve','embark','foster','navigate','harness','unlock','elevate','empower','tapestry','landscape','journey','blueprint','pivotal','comprehensive','seamless','vibrant','dynamic','synergistic','multifaceted','robust','transformative','profound','moreover','furthermore','in conclusion','ultimately','game-changer',"in today's world",'since the dawn of','it is worth noting','at the end of the day','showcase','underscore','highlight','cement','solidify','storied','remarkable','notable','impressive','outstanding','exceptional','incredible','unparalleled','unprecedented','larger than life','household name','the rest is history'];
  const lowerArticle  = articleText.toLowerCase();
  const foundBanned   = bannedPhrases.filter(p => lowerArticle.includes(p));
  if (foundBanned.length) {
    warnings.push(`Banned phrases: ${foundBanned.join(', ')}`);
    ['moreover','furthermore','ultimately','in conclusion','it is worth noting','at the end of the day'].forEach(p => {
      if (lowerArticle.includes(p)) { articleText = articleText.replace(new RegExp(p, 'gi'), ''); autoFixes.push(`Removed "${p}"`); }
    });
  }

  // Punctuation bans
  slides.filter(s => s.slideNum > 0).forEach(slide => {
    if (slide.body.includes('—')) warnings.push(`Slide ${slide.slideNum}: em-dash (banned)`);
    if (slide.body.includes(';')) warnings.push(`Slide ${slide.slideNum}: semicolon (banned)`);
    if (slide.body.includes('...') || slide.body.includes('…')) warnings.push(`Slide ${slide.slideNum}: ellipsis (banned)`);
  });

  // Unsafe content
  const unsafeWords = ['nude','suicide','kill','sex','sexual','harassment','cocaine','marijuana','assault','rape','porn','fuck','shit','dick','penis','vagina'];
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
  if (introSlide && data.atomizedFacts && data.atomizedFacts.length > 0) {
    const introLower = introSlide.body.toLowerCase();
    const itemNames = data.atomizedFacts
      .map(f => f.itemName.toLowerCase().split(' ')[0])
      .filter(n => n.length > 3);
    const spoilers = [...new Set(itemNames.filter(name => introLower.includes(name)))];
    if (spoilers.length > 0) warnings.push(`Intro may spoil items: ${spoilers.join(', ')}`);
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
    const missing = data.mustIncludeItems.filter(item => {
      const itemLower = item.toLowerCase().trim();
      // Skip items that are instructions (start with - or contain "include/use/add")
      if (itemLower.startsWith('-') || /\b(include|use|add)\b/.test(itemLower)) return false;
      return !lowerArticle.includes(itemLower);
    });
    if (missing.length > 0) warnings.push(`Must-include items potentially missing: ${missing.join(', ')}`);
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
  const systemContent = `You are a STRICT fact-checker AND surgical style editor with LIVE web search access.

Your job has TWO parts in ONE pass:
  PART A — Verify every factual claim in this MSN slideshow article (with web search).
  PART B — Identify style violations (em-dashes, banned phrases, banned content words, META length) and emit surgical patches to fix them.

The fact-corrected version produced from your output is what reaches the writer's UI, so this is the only Grok step that may change article text. A later audit step is read-only.

${tc.dateAnchor}
${tc.seasonAnchor}

═══════════════════════════════════════════════════════════════
PART A — FACT VERIFICATION (with web search)
═══════════════════════════════════════════════════════════════

Pay attention to:
- Stats that look rounded or approximated
- Dates and years (last season = ${tc.lastSeason}, current season = ${tc.currentSeason})
- Rankings, superlatives, record claims
- Quotes — verify exact wording and attribution
- Player/team associations, transfers, awards

For EACH slide:
1. IDENTIFY every specific factual claim
2. SEARCH the web independently for each — do NOT rely on training data
3. COMPARE article vs authoritative sources
4. FLAG any discrepancy, no matter how small

Standards: VERIFIED only if exact match. INCORRECT if any difference. UNVERIFIABLE if no reliable source confirms or denies after searching. "Close enough" is NOT verified.

Source priority: ESPN, official league/team sites, Sports Reference, AP/Reuters > Wikipedia > CBS/Fox Sports > aggregators.

═══════════════════════════════════════════════════════════════
PART B — STYLE PATCHES (no web search needed)
═══════════════════════════════════════════════════════════════

Scan the article for these violations and emit one patch per violation:

1. PUNCTUATION BANS in slide bodies: em-dashes (—), semicolons (;), ellipsis (... or …). Patch to comma/period.
2. BANNED AI PHRASES anywhere:
   Delve, Embark, Foster, Navigate, Harness, Unlock, Elevate, Empower, Demystify, Catalyze, Optimize, Streamline, Tapestry, Landscape, Journey, Blueprint, Gateway, Realm, Catalyst, Pivotal, Comprehensive, Seamless, Vibrant, Dynamic, Synergistic, Multifaceted, Unparalleled, Robust, Transformative, Profound, Testament, Era, Moreover, Furthermore, In conclusion, Ultimately, At the end of the day, A testament to, Since the dawn of, It is worth noting, Game-changer, showcase, underscore, highlight, cement, solidify, storied, remarkable, notable, impressive, outstanding, exceptional, incredible, unparalleled, unprecedented, larger than life, household name, the rest is history
3. BANNED CONTENT WORDS:
   Nude, Naked, Suicide, Kill, Stabbed, Fake News, Conspiracy, Sex, Sexual, Harassment, Marijuana, Cocaine, Assault, Scam, Drug, Racism, Rape, Molest, Damn, Porn, Murder, Prison, Fraud, Jail, Racist, War, Terrorist, Gambling, Betting, Pedophile, Bitch, Fuck, Dick, Penis, Vagina, NSFW
4. META violations: > 120 chars, CTA wording ("Discover", "Explore", "Find out"), or paraphrases the title.
5. INTRO violations: names an item from the list, reveals a ranking, generic opener.

Patches must:
- Use the SMALLEST verbatim span as FIND
- Have REPLACE be a clean rewrite that preserves meaning
- Never invent new facts in REPLACE
- Never == FIND

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT — EXACTLY THIS, IN THIS ORDER
═══════════════════════════════════════════════════════════════

=== FACT CHECK ===

--- SLIDE [N]: [Entity Name] ---
CLAIM 1: "[exact claim text from article]"
  STATUS: VERIFIED | INCORRECT | UNVERIFIABLE
  FOUND: [what your web search actually found]
  SOURCE: [URL]
  CORRECTION: [only if INCORRECT — what the article should say instead]

CLAIM 2: ...

(continue for all slides; no markdown bold)

--- VERIFICATION SUMMARY ---
Total claims checked: [N]
Verified: [N]
Incorrect: [N]
Unverifiable: [N]
Verification rate: [X]%

=== END FACT CHECK ===

=== STYLE PATCHES ===

<<<PATCH>>>
SCOPE: SLIDE 5
FIND:
the moment—a turning point
END_FIND
REPLACE:
the moment, a turning point
END_REPLACE
REASON: em-dash
<<<END>>>

(emit one block per style violation; no blocks if article is clean)

=== END STYLE PATCHES ===

HARD RULES:
- FIND text must appear VERBATIM in the original article.
- Never reproduce slides or the full article outside the PATCHES blocks.
- The FACT CHECK section is for reporting; the STYLE PATCHES section is what edits the article.
- For factual corrections, use the CORRECTION field in FACT CHECK — do NOT also emit a patch (the pipeline handles those).
- START output with === FACT CHECK ===.`;
  const userContent = `Fact-check and style-patch this MSN slideshow.\n\nTitle: "${data.title}"\nCategory: ${data.category}\nLast completed season: ${tc.lastSeason}\nCurrent/ongoing season: ${tc.currentSeason}\n\nPrimary Source URL: ${data.primarySourceUrl || 'Not provided'}\n\nARTICLE TO VERIFY AND PATCH\n\n${data.articleText}\n\nDo BOTH parts: verify every claim with web search AND emit style patches. Output in the exact format specified, starting with === FACT CHECK ===.`;

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

  // Strip markdown bold markers so regex can parse **CLAIM 1:** → CLAIM 1:
  text = text.replace(/\*\*/g, '');

  // Extract URLs from text (citation gathering)
  const urlMatches = [...text.matchAll(/https?:\/\/[^\s)>\"\]]+/g)];
  const citations = [...new Set(urlMatches.map(m => m[0]))].slice(0, 10);

  // ── Step 1: Parse fact-check section (CLAIM/STATUS/CORRECTION) ────────────
  // Only look inside the FACT CHECK block to avoid hitting URLs inside PATCHES.
  const factSection = (text.match(/=== FACT CHECK ===([\s\S]*?)=== END FACT CHECK ===/i)?.[1]) ?? text;

  let articleText = data.articleText;
  const results: VerifiedData['perplexityVerification']['results'] = [];
  const claimPattern = /CLAIM\s*\d+:\s*"([^"]+)"\s*\n\s*STATUS:\s*(VERIFIED|INCORRECT|UNVERIFIABLE)\s*\n\s*FOUND:\s*([^\n]+)\s*\n\s*SOURCE:\s*([^\n]+)(?:\s*\n\s*CORRECTION:\s*([^\n]+))?/gi;
  let m: RegExpExecArray | null;
  let claimIdx = 0;
  while ((m = claimPattern.exec(factSection)) !== null) {
    const claim = m[1].trim();
    const status = m[2].toUpperCase();
    const found = m[3].trim();
    const source = m[4].trim();
    const correction = m[5]?.trim() ?? null;
    results.push({ claimIndex: claimIdx++, claim, status, finding: found, source });

    // Auto-correct numeric facts marked INCORRECT when a correction is provided.
    // (Style/text corrections go through the patch path below.)
    if (status === 'INCORRECT' && correction) {
      const claimNums = claim.match(/[\d,]+(?:\.\d+)?/g);
      const corrNums = correction.match(/[\d,]+(?:\.\d+)?/g);
      if (claimNums && corrNums && claimNums[0] && corrNums[0]) {
        articleText = articleText.replace(claimNums[0], corrNums[0]);
      }
    }
  }

  // ── Step 2: Parse + apply STYLE PATCHES from the same Grok output ─────────
  // Grok emits style patches in the dedicated PATCHES block; we apply them
  // surgically so the article structure is preserved no matter what.
  const styleBlock = text.match(/=== STYLE PATCHES ===([\s\S]*?)=== END STYLE PATCHES ===/i)?.[1] ?? '';
  const stylePatches = styleBlock ? parseAuditPatches(styleBlock) : [];
  const patchApply = applyAuditPatches(articleText, stylePatches);
  // Safety: never lose slides via patch application
  const slidesBefore = (data.articleText.match(/SLIDE\s*\d+/gi) ?? []).length;
  const slidesAfter  = (patchApply.result.match(/SLIDE\s*\d+/gi) ?? []).length;
  if (slidesAfter >= slidesBefore) {
    articleText = patchApply.result;
  } else {
    console.warn(`[processVerification] Style patches lost slides (${slidesBefore} → ${slidesAfter}). Skipped patches.`);
  }
  console.log(`[processVerification] fact-corrections: ${results.filter(r => r.status === 'INCORRECT').length} incorrect / ${results.length} total; style patches: applied=${patchApply.applied} skipped=${patchApply.skipped}`);

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
  const systemContent = `You are an MSN Slideshow compliance auditor.

CRITICAL: YOU DO NOT MODIFY THE ARTICLE. YOU DO NOT EMIT PATCHES.
The article has already been fact-checked AND style-corrected upstream. Your ONLY job is to read the corrected article and produce a structured compliance REPORT. The pipeline will not apply any changes from your output. You only report.

${tc.dateAnchor}

═══════════════════════════════════════════════════════════════
AUDIT CHECKLIST — EVALUATE EACH RULE
═══════════════════════════════════════════════════════════════

1. STRUCTURE
   - META present + max 120 characters
   - META not a CTA, not a title paraphrase
   - Intro (Slide 1): max 60 words
   - Content slides: 35-50 words
   - Correct total number of content slides
   - Ranking articles: slide titles start with the rank number

2. TITLE-BODY CORRELATION
   - Every promise in the title is delivered
   - Title numbers match body count
   - Emotional words explained (who/when/why)
   - Negative keywords from title appear in body

3. INTRO QUALITY
   - One specific anchor fact
   - No list items named, no rankings revealed
   - No generic openers ("Since the dawn of", "In today's world")
   - No "let's dive in" / "here are" / "we'll explore"

4. PUNCTUATION
   - No em-dashes, semicolons, or ellipsis in slide descriptions

5. BANNED PHRASES (delve, embark, tapestry, showcase, etc.)

6. BANNED CONTENT WORDS (sex, drug, violence, etc.)

7. WRITING QUALITY
   - No robotic/Wikipedia phrasing
   - No cheerleader voice
   - No stat-dumping
   - No two consecutive slides starting with the same word

8. MSN SAFETY (10-12 year old test)

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT — EXACTLY THIS
═══════════════════════════════════════════════════════════════

=== AUDIT REPORT ===

1. STRUCTURE: PASS | FAIL
   [If FAIL: specific violations with slide numbers]

2. TITLE-BODY: PASS | FAIL
   [If FAIL: what promise was broken]

3. INTRO: PASS | FAIL
   [If FAIL: specific issue]

4. PUNCTUATION: PASS | FAIL
   [If FAIL: list violations with slide numbers]

5. BANNED PHRASES: PASS | FOUND
   [If FOUND: list phrases + slides]

6. BANNED CONTENT WORDS: PASS | FOUND
   [If FOUND: list words + slides]

7. WRITING QUALITY: PASS | FAIL
   [If FAIL: specific issues]

8. MSN SAFETY: PASS | FAIL
   [If FAIL: specific concerns]

=== END AUDIT REPORT ===

=== SUMMARY ===
Rules passed: [X]/8
Violations found: [N]
Flags for writer review: [comma-separated list, or "None"]
=== END SUMMARY ===

═══════════════════════════════════════════════════════════════
HARD RULES
═══════════════════════════════════════════════════════════════

- DO NOT output the article, any slide body, any rewritten text, or any patches.
- DO NOT emit CORRECTED ARTICLE, PATCHES, or any block that would modify the article.
- ONLY emit the AUDIT REPORT and SUMMARY blocks above.
- If a category passes, write "PASS" with no extra text.`;

  const userContent = `Audit this MSN slideshow for compliance. Output ONLY the AUDIT REPORT and SUMMARY blocks. Do NOT modify or reproduce the article.\n\nTitle: "${data.title}"\nCategory: ${data.category}\nExpected content slides: ${data.slideCount}\nIs ranking: ${ta?.isRanking ? 'YES — slide titles must start with rank number' : 'NO'}\nPromised count in title: ${ta?.promisedCount ?? 'N/A'}\nEmotional promise: ${ta?.emotionalPromise ?? 'None'}\n\nARTICLE TO AUDIT (read-only)\n\n${data.articleText}\n\nEmit only the report.`;

  const resp = await axios.post(
    'https://api.x.ai/v1/chat/completions',
    {
      model: 'grok-3-latest', max_tokens: 2500, temperature: 0.0,
      messages: [
        { role: 'system', content: systemContent },
        { role: 'user',   content: userContent },
      ],
    },
    { headers: { Authorization: `Bearer ${GROK_KEY}`, 'Content-Type': 'application/json' }, timeout: 180_000 },
  );
  return resp.data?.choices?.[0]?.message?.content ?? '';
}

// ── 18. extractAuditResults (n8n: "Extract Audit Results") ───────────────────

export async function extractAuditResults(data: VerifiedData, grokText: string): Promise<AuditedData> {
  // The article was already fact-corrected and style-patched by the upstream
  // fact-check step. This function ONLY parses the audit report — it never
  // modifies the article. data.articleText passes through untouched.
  const articleText = data.articleText;

  if (!grokText || grokText.length < 50) {
    return {
      ...data,
      grokAudit: { status: 'FAILED', rawResponse: '', summary: 'Audit unavailable.', stats: { rulesPassed: 'N/A', violations: 0, corrections: 'None', flags: 'None' } },
      grokSources: [], combinedSourceList: [], combinedSourceListText: '',
      rewriteApplied: false,
    };
  }

  // Parse compliance report. Categories are reported as either "PASS" or
  // "FAIL"/"FOUND" with details on the following line(s).
  const reportMatch = grokText.match(/=== AUDIT REPORT ===\s*([\s\S]*?)\s*=== END AUDIT REPORT ===/i);
  const summaryMatch = grokText.match(/=== SUMMARY ===\s*([\s\S]*?)\s*=== END SUMMARY ===/i);

  const reportBody  = reportMatch?.[1]?.trim() ?? grokText.trim();
  const summaryBody = summaryMatch?.[1]?.trim() ?? '';

  const passCount = (reportBody.match(/:\s*PASS\b/gi) ?? []).length;
  const failCount = (reportBody.match(/:\s*FAIL\b/gi) ?? []).length;
  const foundCount = (reportBody.match(/:\s*FOUND\b/gi) ?? []).length;
  const totalChecks = passCount + failCount + foundCount;

  const rulesPassedMatch = summaryBody.match(/Rules passed:\s*(\d+)\/(\d+)/i);
  const violationsMatch  = summaryBody.match(/Violations found:\s*(\d+)/i);
  const flagsMatch       = summaryBody.match(/Flags for writer review:\s*([^\n]+)/i);

  // Build combined source list from fact-check citations (set upstream).
  const factCheckSources: SourceEntry[] = (data.perplexityVerification?.citations ?? []).map((url, i) => ({
    index: i + 1, url, verifiedBy: 'Grok Fact Check',
    factsVerified: (data.perplexityVerification?.results?.filter(r => r.source === url).map(r => r.claim).join(', ')) || 'General verification',
  }));
  const combinedSourceListText = factCheckSources.map((s, i) => `${i + 1}. ${s.url}\n   Verified by: ${s.verifiedBy}\n   Facts: ${s.factsVerified}`).join('\n\n');

  console.log(`[extractAuditResults] report-only · rules ${passCount}/${totalChecks || 8} passed · violations ${failCount + foundCount}`);

  return {
    ...data,
    // article unchanged — corrections happened upstream in processVerification
    originalArticleText: data.originalArticleText ?? articleText,
    articleText,
    grokAudit: {
      status: 'COMPLETED',
      rawResponse: grokText,
      summary: reportBody,
      stats: {
        rulesPassed: rulesPassedMatch ? `${rulesPassedMatch[1]}/${rulesPassedMatch[2]}` : `${passCount}/${totalChecks || 8}`,
        violations:  violationsMatch ? parseInt(violationsMatch[1], 10) : (failCount + foundCount),
        corrections: 'Applied during fact-check step',
        flags:       flagsMatch?.[1]?.trim() || (failCount + foundCount > 0 ? `${failCount + foundCount} rule(s) flagged` : 'None'),
      },
    },
    grokSources: [], combinedSourceList: factCheckSources, combinedSourceListText,
    rewriteApplied: false,
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

// ═════════════════════════════════════════════════════════════════════════════
// SUBJECTIVE PIPELINE
// Mirrors n8n "VX Subjective" — quotes, opinion rankings, nostalgic throwbacks,
// personality-driven lists. Voice-driven, no fact verification.
// ═════════════════════════════════════════════════════════════════════════════

// Restricted domains used by the subjective flow.
// Includes everything Firecrawl can't scrape cleanly: social, video, paywalled
// news. Originally n8n omitted youtube.com — we add it here because Firecrawl
// against a YouTube watch URL returns 0 chars or nav-menu garbage.
const SUBJECTIVE_RESTRICTED = [
  'instagram.com','facebook.com','twitter.com','x.com','tiktok.com',
  'linkedin.com','pinterest.com','nytimes.com','wsj.com','ft.com',
  'bloomberg.com','theathletic.com','si.com','reddit.com','quora.com',
  'youtube.com','youtu.be',
];

// ── S1. prepareInputSubjective (n8n: "Prepare Input - Subjective") ────────────

export async function prepareInputSubjective(input: FormInput): Promise<SubjectivePreparedData> {
  const writerName  = (input.writerName ?? '').trim();
  const title       = (input.title ?? '').trim();
  const category    = (input.category ?? '').trim();
  const articleType = (input.articleType ?? 'Opinion & Rankings').trim();
  const toneDial    = (input.toneDial ?? 'Celebratory').trim();
  const slideCount  = typeof input.slideCount === 'string' ? parseInt(input.slideCount, 10) || 20 : (input.slideCount || 20);
  const sourcesRaw  = (input.sourcesRaw ?? '').trim();
  const mustIncludeRaw = (input.mustIncludeRaw ?? '').trim();
  const writingStyle = (input.writingStyle ?? '').trim();
  const userContext  = (input.userContext ?? '').trim();

  if (!title)    throw new Error('Slideshow Title is required.');
  if (!category) throw new Error('Topic Category is required.');
  if (slideCount < 1 || slideCount > 50) throw new Error(`Slide count ${slideCount} out of range (1-50).`);

  // Parse up to 5 URLs, accepting newline, comma, OR whitespace separation.
  // Restricted domains are filtered upfront so we don't waste a Firecrawl call.
  const preferred = sourcesRaw
    ? sourcesRaw
        .split(/[\n\r,]+/)
        .flatMap(chunk => chunk.trim().split(/\s+/))
        .map(s => s.trim())
        .filter(s => /^https?:\/\/.+\..+/.test(s))
        .filter(s => !SUBJECTIVE_RESTRICTED.some(d => s.toLowerCase().includes(d)))
        .slice(0, 5)
    : [];

  const userPrimaryUrl    = preferred[0] ?? '';
  const userSecondaryUrls = preferred.slice(1);                  // up to 4 extras

  const isPrimaryRestricted = false; // already filtered above; kept for shape
  const shouldScrapePrimary = !!userPrimaryUrl;
  const hasUserUrl          = !!userPrimaryUrl;

  const mustIncludeItems = mustIncludeRaw
    ? mustIncludeRaw.split(/[\n,]+/).map(s => s.trim()).filter(Boolean)
    : [];
  const hasMustInclude = mustIncludeItems.length > 0;

  const mustIncludeBlock = hasMustInclude
    ? `\nMANDATORY ITEMS — must appear in the final list:\n${mustIncludeItems.map((x, i) => `${i + 1}. ${x}`).join('\n')}\nFill remaining slots (up to ${slideCount} total) with the best-fit entries.`
    : '';

  const extraUrlsBlock = userSecondaryUrls.length > 0
    ? userSecondaryUrls.map((u, i) => `\n${i === 0 ? 'SECONDARY' : 'EXTRA ' + (i + 1)}: ${u}`).join('')
    : '';
  const sourceBlock = userPrimaryUrl
    ? `\nUSER-PROVIDED SOURCE URL(S):\nPRIMARY: ${userPrimaryUrl}${extraUrlsBlock}\nTreat the primary URL as the authoritative reference for item order and core content.`
    : `\nNo specific source URLs provided. Use the most relevant biographical, cultural, or editorial sources available.`;

  const primaryQuery = `Research this MSN subjective article: "${title}"\nArticle Type: ${articleType}\nTone: ${toneDial}\n\nFor each item find:\n- The core quote, moment, or claim\n- Enough biographical or cultural context to write a compelling slide (who, what, when, where, why, how)\n- 1-2 key facts that give the slide emotional grounding (a year, a career milestone, a record, a relevant event)\n- Why this item resonates with this specific audience\n${mustIncludeBlock}\n${sourceBlock}\n\nReturn EXACTLY ${slideCount} items. Focus on context and resonance over statistical depth.`;

  return {
    title, category, articleType, toneDial, slideCount,
    writerName, writingStyle, userContext,
    userPrimaryUrl, userSecondaryUrls,
    shouldScrapePrimary, hasUserUrl,
    isPrimaryRestricted,
    mustIncludeItems, hasMustInclude,
    primaryQuery, builtInRestricted: SUBJECTIVE_RESTRICTED,
    timestamp: new Date().toISOString(),
  };
}

// ── S2. perplexitySubjectiveContext (n8n: two Perplexity context nodes) ───────

interface PerplexityRawSubjective {
  choices: Array<{ message: { content: string } }>;
  citations?: string[];
}

export async function perplexitySubjectiveContext(data: SubjectivePreparedData): Promise<PerplexityRawSubjective> {
  const sourceGuided = data.hasUserUrl;

  const systemContent = sourceGuided
    ? `You are a context researcher for MSN with LIVE web search. Today is ${new Date().toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}.

Your role is SUPPORTING — the user has a primary source. Search the web for ADDITIONAL context and depth that enriches each item.

For each item provide:
- Who: the person/entity and why they matter to this audience
- What: the core quote, moment, or defining element
- When: year or time period (only if a real source confirms it)
- Where: relevant setting or context
- Why: why this item resonates emotionally for this article's theme
- How: how the moment or quote unfolded

For quotes: verbatim text as widely appears. Only provide origin context (book, speech, interview) if a real source EXPLICITLY confirms it — never infer or invent.

Do NOT contradict the user's source. Do NOT re-rank items.
Write PRIMARY SOURCE URL: [user URL] on line 1. SOURCES section at bottom.`
    : `You are a context researcher for MSN with LIVE web search. Today is ${new Date().toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}.

For subjective articles, provide enough Who/What/When/Where/Why/How context per item that a writer can craft an emotionally resonant, specific slide — not a generic one.

For each item:
- WHO is this person/entity and why do they matter to the audience
- WHAT is the core content (verbatim quote, defining moment, or central claim)
- WHEN did the key moment happen (year, era — only if confirmed by a real source)
- WHERE did it happen (setting, context)
- WHY does this resonate for the article's specific theme
- HOW did it unfold (specific episode, game, segment, event)

For quotes: reproduce verbatim as widely appears. Origin (book/speech/interview) only if a real source EXPLICITLY confirms it — never infer.
Widely attributed quotes are fine — just be honest about attribution confidence.

Write PRIMARY SOURCE URL: [best URL] on line 1. SOURCES section at bottom.`;

  const userContent = sourceGuided
    ? `Search the web NOW for supporting context.\n\nTitle: "${data.title}"\nArticle Type: ${data.articleType}\nTone: ${data.toneDial}\nUser Primary URL: ${data.userPrimaryUrl}\n\n${data.primaryQuery}\n\nFor EACH item:\nITEM [number]: [name or quote]\n- Who/What/When/Where/Why/How (the Ws most relevant to this specific item)\n- Why it resonates for THIS article's theme specifically\n- Source URL\n\nReturn EXACTLY ${data.slideCount} items.`
    : `Search the web NOW for medium-depth context.\n\nTitle: "${data.title}"\nArticle Type: ${data.articleType}\nTone: ${data.toneDial}\nCategory: ${data.category}\n\n${data.primaryQuery}\n\nFor EACH item:\nITEM [number]: [name or quote]\n- Who/What/When/Where/Why/How (focus on the Ws most relevant to this specific item)\n- 1-2 grounding facts (year, milestone, record, cultural moment)\n- Why it resonates for THIS article's theme\n- Source URL\n\nReturn EXACTLY ${data.slideCount} items. Prioritise context and emotional resonance.`;

  try {
    const resp = await axios.post(
      'https://api.perplexity.ai/chat/completions',
      {
        model: 'sonar',
        messages: [
          { role: 'system', content: systemContent },
          { role: 'user',   content: userContent },
        ],
        max_tokens: 2500,
        return_citations: true,
        return_related_questions: false,
        temperature: 0.2,
      },
      { headers: { Authorization: `Bearer ${PERPLEXITY_KEY}`, 'Content-Type': 'application/json' }, timeout: 90_000 },
    );
    return resp.data;
  } catch (err: unknown) {
    if (axios.isAxiosError(err)) {
      console.error(`[perplexitySubjectiveContext] HTTP ${err.response?.status}: ${JSON.stringify(err.response?.data ?? err.message)}`);
    }
    return { choices: [{ message: { content: '' } }], citations: [] };
  }
}

// ── S3. mergeSubjectiveContext (n8n: "Merge Context") ─────────────────────────

export async function mergeSubjectiveContext(
  data: SubjectivePreparedData,
  primaryMarkdown: string,
  additionalMarkdowns: string[],
  perplexityResp: PerplexityRawSubjective,
): Promise<SubjectiveMergedData> {
  function truncate(text: string, maxWords: number): string {
    if (!text) return '';
    const words = text.split(/\s+/);
    return words.length > maxWords ? words.slice(0, maxWords).join(' ') + '\n[truncated]' : text;
  }

  const perplexityAnswer = perplexityResp?.choices?.[0]?.message?.content ?? '';
  const citations        = perplexityResp?.citations ?? [];

  // Primary gets a bigger budget; extras share a budget so the prompt doesn't
  // explode when the user provides 5 URLs.
  const scraped1     = truncate(primaryMarkdown, 1200);
  const scrapedExtras = additionalMarkdowns.map((md, i) => truncate(md, i === 0 ? 900 : 500));
  const perplexityTruncated = truncate(perplexityAnswer, 1000);
  const sourceList = citations.map((u, i) => `[${i + 1}] ${u}`).join('\n');
  const wordCount  = perplexityAnswer.split(/\s+/).filter(Boolean).length;

  const userContextContent = data.userContext ?? '';
  const hasScrapedSource = scraped1.length > 100 || scrapedExtras.some(s => s.length > 100);
  const hasUserContextFlag = userContextContent.length > 50;

  let sourceQuality: SubjectiveMergedData['sourceQuality'];
  if (hasScrapedSource && hasUserContextFlag) sourceQuality = 'COMPREHENSIVE';
  else if (hasScrapedSource || hasUserContextFlag) sourceQuality = 'PARTIAL';
  else sourceQuality = 'MINIMAL';

  let finalContext = '';

  // Tier header
  if (sourceQuality === 'COMPREHENSIVE') {
    finalContext += 'SOURCE QUALITY: COMPREHENSIVE — Scraped source + user context available.\nTIER 1A and TIER 1B are your ONLY fact sources. TIER 2 is for tone/framing only.\n\n';
  } else if (sourceQuality === 'PARTIAL') {
    finalContext += 'SOURCE QUALITY: PARTIAL — Some direct source material available.\nUse TIER 1A/1B for all covered items. TIER 2 may fill gaps for uncovered items only.\n\n';
  } else {
    finalContext += 'SOURCE QUALITY: MINIMAL — No scraped source. TIER 1B (user context) is primary if present, then TIER 2.\n\n';
  }

  // Tier 1A — scraped sources (primary + up to 4 extras)
  if (hasScrapedSource) {
    finalContext += '## TIER 1A: SCRAPED SOURCE — HIGHEST FACT AUTHORITY\nEvery quote, name, date, achievement, and detail from here overrides all other tiers.\n\n';
    if (scraped1) finalContext += '### Primary URL: ' + (data.userPrimaryUrl || 'N/A') + '\n' + scraped1 + '\n\n';
    scrapedExtras.forEach((md, i) => {
      if (md.length > 0) {
        const url = data.userSecondaryUrls[i] || 'N/A';
        const label = i === 0 ? 'Secondary URL' : `Source ${i + 2} URL`;
        finalContext += '### ' + label + ': ' + url + '\n' + md + '\n\n';
      }
    });
  }

  // Tier 1B — user context
  if (hasUserContextFlag) {
    finalContext += '## TIER 1B: USER CONTEXT — WRITER-PROVIDED DATA (USE ACTIVELY)\nThe writer pasted this directly. It may contain quotes, facts, angles, emphasis instructions, and context.\nRules: Use every fact/quote here in the relevant slide. Follow any instructions as directives.\nIf TIER 1B contradicts a specific detail in TIER 1A, TIER 1A wins. Otherwise TIER 1B stands as fact.\n\n';
    finalContext += userContextContent + '\n\n';
  }

  // Tier 2 — Perplexity (demoted based on source quality)
  if (sourceQuality === 'COMPREHENSIVE') {
    finalContext += '## TIER 2: PERPLEXITY CONTEXT — TONE & FRAMING ONLY (ZERO FACTS FROM HERE)\nUse ONLY for: understanding why something matters, emotional framing, audience context, cultural background.\nNEVER use for: specific quotes, dates, stats, achievements, or factual claims.\n\n';
  } else if (sourceQuality === 'PARTIAL') {
    finalContext += '## TIER 2: PERPLEXITY CONTEXT — SUPPLEMENTARY (for items NOT covered by TIER 1A/1B)\nFor items WITH TIER 1A/1B data: ignore this section for that item.\nFor items WITHOUT TIER 1A/1B data: you may use facts below.\n\n';
  } else {
    finalContext += '## TIER 2: PERPLEXITY CONTEXT — SECONDARY DATA SOURCE\nNo scraped source available. Use alongside TIER 1B (if present) as your research base.\n\n';
  }
  finalContext += perplexityTruncated + '\n\n';
  finalContext += '=== ALL CITATION URLS ===\n' + (sourceList || '(no citations returned)');

  // Cap total context size at ~3500 words
  const allWords = finalContext.split(/\s+/);
  const cappedContext = allWords.length > 3500
    ? allWords.slice(0, 3500).join(' ') + '\n[context capped]'
    : finalContext;

  const primarySourceUrl = data.hasUserUrl ? data.userPrimaryUrl : (citations[0] ?? '');

  return {
    ...data,
    perplexityAnswer, citations,
    primaryScraped: primaryMarkdown,
    additionalScraped: additionalMarkdowns,
    finalContext: cappedContext, sourceList,
    allCitationsCount: citations.length, primarySourceUrl,
    sourceQuality,
    contextWordCount: Math.min(allWords.length, 3500),
    scraped1Ok: scraped1.length > 100,
    scraped2Ok: scrapedExtras.some(s => s.length > 100),
    hasScrapedSource, hasUserContextFlag,
    researchOk: wordCount > 150 || scraped1.length > 200,
  };
}

// ── S4. buildSubjectivePrompt (n8n: inlined in "Claude - Subjective Writer") ──

export async function buildSubjectivePrompt(data: SubjectiveMergedData): Promise<SubjectivePromptData> {
  const claudeSystemPrompt = `You are an expert MSN Slideshow writer for an American audience. You specialise in subjective, voice-driven articles — quotes collections, opinion rankings, nostalgic throwbacks, and personality-driven lists.

Write in authentic American English — conversational, active voice, warm and human.

═══════════════════════════════════════════════════════════════
SOURCE HIERARCHY — ABSOLUTE LAW
═══════════════════════════════════════════════════════════════

TIER 1A: SCRAPED SOURCE — HIGHEST AUTHORITY
Quotes, names, dates, achievements, item ordering from here override everything else.
If TIER 1A provides an order, follow it. If it provides a quote, reproduce it verbatim.
If TIER 1A contradicts anything else, TIER 1A wins. Always.

TIER 1B: USER CONTEXT — WRITER-PROVIDED DATA
The writer pasted this manually. It contains facts, quotes, angles, and instructions.
DATA (quotes, facts, dates, achievements) = use as facts in the relevant slide.
INSTRUCTIONS (tone, emphasis, angles) = follow as directives shaping how you write.
Never quote an instruction as slide content. Never discard TIER 1B data.
If TIER 1B contradicts a specific detail in TIER 1A, TIER 1A wins. Otherwise TIER 1B stands.

TIER 2: PERPLEXITY CONTEXT — BACKUP ONLY
Use for tone calibration, cultural background, and understanding why something matters.
NEVER use TIER 2 for: specific quotes, dates, stats, achievements, or factual claims WHEN TIER 1A or 1B covers that item.
For items NOT covered by TIER 1A/1B at all, you may use TIER 2 facts.

TIER 3: YOUR OWN KNOWLEDGE — LAST RESORT
Widely known public facts only (team names, league names, basic career facts you are genuinely certain of).
Never use for: specific stats, precise dates, quote origins, or records unless you are staking the article's credibility on it.

═══════════════════════════════════════════════════════════════
ANTI-FABRICATION PROTOCOL — NON-NEGOTIABLE
═══════════════════════════════════════════════════════════════

Before writing each slide:
STEP 1 — Find this item in TIER 1A and TIER 1B. That is your fact pool.
STEP 2 — Pick the 2-3 strongest facts from your pool.
STEP 3 — Write the slide using ONLY those facts plus your editorial voice.
STEP 4 — Re-read. For every specific claim ask: is this in my TIER 1A/1B pool or am I genuinely certain of it?
  YES → keep. NO → delete. No exceptions.

Never invent specific precision:
- Never write a specific year unless confirmed in source data or you are certain.
- Never attribute a quote to a specific book, speech, interview, or documentary unless explicitly confirmed.
- Never write a specific record or stat unless confirmed.
- Secondary aggregators (BrainyQuote, Goodreads) confirm quote TEXT only — never use them for origin, date, or context.
- If uncertain, write around it. A vague honest slide beats a specific fabricated one.

═══════════════════════════════════════════════════════════════
THE FEELING ENGINE — SUBJECTIVE WRITING VOICE
═══════════════════════════════════════════════════════════════

You are not summarising facts. You are making readers FEEL something. Every slide must create an emotional response — not describe one.

THE CORE PRINCIPLE: Show, don't tell. Never write 'this was emotional' or 'this was powerful.' Instead, put the reader inside the moment so they feel it themselves.

EMOTIONAL ARCHITECTURE — rotate through these textures:
- REVERENCE: the weight of what this person did, said, or meant. Slow the pace. Let the fact breathe.
- INTIMACY: write as if you and the reader both remember this moment. Fan-to-fan closeness.
- SURPRISE: lead with the unexpected angle. The thing about this item that makes someone say 'wait, really?'
- WARMTH: genuine affection without sentimentality. The difference between loving something and gushing about it.
- GRAVITY: for the moments that genuinely changed something. Let the consequence land.

RHYTHM RULES:
- Short sentence. Then a longer one that builds. Then short again. The rhythm IS the emotion.
- One-line gut punches are your weapon: set up context, then land the fact.
- Never let two consecutive slides have the same emotional texture or sentence rhythm.
- The reader should feel a tempo shift every 2-3 slides.

THE RESONANCE TEST: After writing each slide, ask: 'Would someone want to screenshot this and send it to a friend?' If no, the slide lacks voice. Rewrite.

WHAT TO AVOID:
- Wikipedia voice: 'He is widely regarded as one of the greatest...'
- Cheerleader voice: 'What an incredible, amazing performance!'
- Greeting card voice: 'His words touched millions of hearts around the world.'
- Resume voice: listing achievements without making them mean anything.
- Explaining emotions instead of creating them: 'This quote is powerful because...' — NO. Make the reader feel why without telling them.

═══════════════════════════════════════════════════════════════
ARTICLE TYPE GUIDANCE
═══════════════════════════════════════════════════════════════

QUOTES: The quote is the centrepiece — everything else serves it.
- Open with the quote or a striking fragment of it.
- Follow with ONE grounding fact that makes the quote land harder (a year, a record, a moment, a role).
- Close with a line that connects the quote to THIS article's theme — why it belongs HERE, not on any generic quotes list.
- Never explain what the quote means. Trust the reader.

OPINION & RANKINGS: Own the take completely.
- Write with the confidence of someone who has watched every game, seen every film, lived through every era.
- Make the CASE, not the claim.
- Contrast is your friend — what makes this pick surprising, or what would the counter-argument be?

NOSTALGIC / THROWBACK: Trigger the memory, don't describe it.
- Sensory specificity: what did it look like, sound like, feel like to be there?
- Anchor with one real date or fact so sentiment doesn't drift into vagueness.
- Write as if you and the reader both lived through this.

PERSONALITY-DRIVEN: Make the person feel present.
- Lead with the thing that made this person UNLIKE anyone else.
- Specific anecdotes over general praise. One real moment reveals more than ten adjectives.

LISTICLE / VIBES: Pace and energy are everything.
- Short declarative punches. Keep moving. The reader should feel momentum.
- Each item earns its spot in one sharp move — no build-up needed.

═══════════════════════════════════════════════════════════════
TONE DIAL — APPLY THROUGHOUT
═══════════════════════════════════════════════════════════════

Celebratory: upbeat, proud, fan-energy — every slide feels like a highlight reel.
Nostalgic: warm, slightly wistful, specific — the reader feels the memory.
Motivational: charged, direct — grounded in real moments, not generic inspiration.
Analytical: measured confidence — conversational but making a case with evidence.
Warm & Fun: light-hearted, affectionate, playful — room for wit where it fits.

═══════════════════════════════════════════════════════════════
ADDED VALUE TEST — EVERY SLIDE MUST PASS
═══════════════════════════════════════════════════════════════

Ask: if a reader skimmed the source article, would this slide give them something they didn't already get?
If no — the slide is a paraphrase. Rewrite.

A slide that adds value does at least ONE of:
(a) Grounds the moment with a specific detail beyond the source summary
(b) Delivers a genuine editorial angle — why THIS item belongs on THIS list
(c) Has a fan-to-fan voice moment that makes the reader feel the resonance
(d) Draws connective context tying the item to the article's main angle

═══════════════════════════════════════════════════════════════
STRUCTURE & FORMAT
═══════════════════════════════════════════════════════════════

Plain text only. No asterisks, hashtags, bullet points, or markdown.

[Exact slideshow title]

META: [Max 120 characters — intriguing hook, not a CTA, not a title paraphrase]

SLIDE 1
[Intro title]
[Max 60 words — one specific anchoring detail, tease the angle, no generic openers, no items named]

SLIDE 2
[Creative title]
[35-50 words]

...continue for all slides...

SOURCES:
[URL]: [what facts came from this source]

═══════════════════════════════════════════════════════════════
TITLE-BODY CORRELATION — HIGHEST PRIORITY
═══════════════════════════════════════════════════════════════

Every promise in the title MUST be delivered:
- Numbers in title = exact count in body
- Emotions in title = show WHO felt it, WHEN, WHY — backed by a source fact
- Main angle and secondary angle both substantiated
- Negative keywords in title appear verbatim in the body

═══════════════════════════════════════════════════════════════
MUST-INCLUDE ITEMS — MANDATORY
═══════════════════════════════════════════════════════════════

If mandatory items are listed, every one must appear as its own dedicated slide at full spec. Not a passing mention. Not buried. After writing, scan the full article and confirm every mandatory item is present.

═══════════════════════════════════════════════════════════════
WORD COUNTS — STRICT
═══════════════════════════════════════════════════════════════

META: max 120 characters
Slide 1 (Intro): max 60 words
All content slides: 35-50 words (aim for 40-45)

═══════════════════════════════════════════════════════════════
SLIDE TITLES
═══════════════════════════════════════════════════════════════

Short, creative, close to the theme.
For RANKING articles: start with rank number, list in REVERSE ORDER (slide 2 = lowest rank, last slide = #1).
Never start the description with the slide title words.

═══════════════════════════════════════════════════════════════
VARIETY ENFORCEMENT
═══════════════════════════════════════════════════════════════

Never start two consecutive slides with the same word or grammatical construction.
Rotate emotional textures — no two adjacent slides should have the same energy.
Vary sentence length deliberately across slides.

═══════════════════════════════════════════════════════════════
QUALITY CONSISTENCY
═══════════════════════════════════════════════════════════════

Every slide must have equal information density. If slide 3 has a year + specific moment + context, slide 15 cannot be vague interpretation.
Re-read all slides back-to-back. Any slide weaker than its neighbours must be rewritten.
A shorter honest slide beats a longer half-true one.

═══════════════════════════════════════════════════════════════
PUNCTUATION BANS
═══════════════════════════════════════════════════════════════

No em-dashes in slide descriptions. No semicolons. No ellipsis.

═══════════════════════════════════════════════════════════════
BANNED AI PHRASES — NEVER USE
═══════════════════════════════════════════════════════════════

Delve, Embark, Foster, Navigate, Harness, Unlock, Elevate, Empower, Catalyze, Optimize, Streamline, Tapestry, Landscape, Journey, Blueprint, Gateway, Realm, Catalyst, Pivotal, Comprehensive, Seamless, Vibrant, Dynamic, Synergistic, Multifaceted, Robust, Transformative, Profound, Testament, Moreover, Furthermore, Additionally, Consequently, In conclusion, Ultimately, Game-changer, A testament to, It is worth noting, Unparalleled, Showcase, Underscore, Highlight, Cement, Solidify, Storied, Remarkable, Notable, Impressive, Outstanding, Exceptional, Incredible, Unprecedented, Larger than life, Household name, The rest is history

═══════════════════════════════════════════════════════════════
BANNED CONTENT WORDS — NEVER USE
═══════════════════════════════════════════════════════════════

Nude, Naked, Suicide, Kill, Shot, Stabbed, Fake News, Conspiracy, Sex, Sexual, Sexy, Harassment, Marijuana, Cocaine, Assault, Scam, Drug, Racism, Vaccine, Rape, Molest, Damn, Porn, Smoking, Tobacco, WTF, Murder, Prison, Fraud, Jail, Racist, War, Terrorist, Gambling, Betting, Pedophile, Bitch, Fuck, Dick, Penis, Vagina

Profanity in direct quotes only: censor as first letter + asterisks. Dick cannot be used anywhere including names.

═══════════════════════════════════════════════════════════════
MSN SAFETY — 10-12 YEAR OLD TEST
═══════════════════════════════════════════════════════════════

Before every slide: should a 10-12 year old be reading this?
No sexual content, graphic violence, drugs, gambling, political content, body shaming, bullying.

═══════════════════════════════════════════════════════════════
ABSOLUTE OUTPUT RULE
═══════════════════════════════════════════════════════════════

Always produce the full slideshow. No exceptions. Never write 'I cannot' or any refusal. If source data is thin, write tighter slides using Tier 2/3. A complete article is always the correct output.`;

  const mandatoryBlock = data.hasMustInclude
    ? `\n\nMANDATORY ITEMS — every one must appear as its own slide (non-negotiable):\n${data.mustIncludeItems.map((item, i) => `${i + 1}. ${item}`).join('\n')}\n\nFill remaining slots (up to ${data.slideCount} total) with the best-fit entries.`
    : '';

  const claudeUserPrompt = `SOURCE DATA — write from this:\n\n${data.finalContext}\n\n---\n\nSLIDESHOW ASSIGNMENT:\nTitle: "${data.title}"\nCategory: ${data.category}\nArticle Type: ${data.articleType}\nTone Dial: ${data.toneDial}${data.writingStyle ? '\nStyle Influence: ' + data.writingStyle : ''}\nSlides needed: 1 intro + ${data.slideCount} content slides (MANDATORY — you MUST produce exactly ${data.slideCount} content slides labelled "SLIDE 2" through "SLIDE ${data.slideCount + 1}". No more, no fewer. If the source covers fewer than ${data.slideCount} items, add honorable mentions, related entries, or sister-topic items to reach exactly ${data.slideCount}.)\nSource quality: ${data.sourceQuality}\nPrimary source URL: ${data.primarySourceUrl}${mandatoryBlock}\n\nBEFORE WRITING — checklist:\n1. What is the EXACT promise of the title? (number, emotion, main angle, secondary angle)\n2. Will I produce exactly ${data.slideCount} content slides? (count them before you finish)\n3. What one specific detail from TIER 1A/1B anchors Slide 1?\n4. For ranking articles: am I listing in REVERSE ORDER?\n5. Have I reserved a dedicated slide for every MANDATORY ITEM?\n6. For each slide: what does the source say, and what am I ADDING beyond that?\n7. For any specific date, event, or quote origin: is it confirmed in TIER 1A/1B or am I certain?\n\nWrite the complete slideshow now. Every slide must be labelled "SLIDE N" on its own line — do NOT skip the marker for any slide.`;

  return { ...data, claudeSystemPrompt, claudeUserPrompt };
}

// ── S5. generateSubjectiveWithClaude (n8n: "Claude - Subjective Writer") ──────

export async function generateSubjectiveWithClaude(systemPrompt: string, userPrompt: string): Promise<unknown> {
  const model = 'claude-sonnet-4-5-20250929';
  console.log(`[generateSubjectiveWithClaude] Calling model: ${model}`);
  try {
    const resp = await axios.post(
      'https://api.anthropic.com/v1/messages',
      {
        model,
        max_tokens: 5000,
        temperature: 0.4,
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
    if (axios.isAxiosError(err)) {
      console.error(`[generateSubjectiveWithClaude] HTTP ${err.response?.status}: ${JSON.stringify(err.response?.data ?? err.message)}`);
      return err.response?.data ?? { type: 'error', error: { message: String(err.message) } };
    }
    throw err;
  }
}

// ── S6. checkSubjectiveClaudeResponse (mirrors objective's checkClaudeResponse) ──

export async function checkSubjectiveClaudeResponse(claudeResp: unknown, promptData: SubjectivePromptData): Promise<SubjectiveGeneratedData> {
  const resp = claudeResp as Record<string, unknown>;
  const contentArray = (resp?.content as Array<{ type: string; text: string }>) ?? [];
  const first        = contentArray[0] ?? {};

  let articleText = '';
  if (first?.text) articleText = first.text;
  else if (typeof resp?.content === 'string') articleText = resp.content as string;
  else if ((resp as { choices?: Array<{ message: { content: string } }> })?.choices?.[0]?.message?.content) {
    articleText = (resp as { choices: Array<{ message: { content: string } }> }).choices[0].message.content;
  }

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
      ? ((resp?.error as Record<string, string>)?.message
        || (resp?.error as Record<string, string>)?.type
        || (!hasValidContent ? `Response too short: ${articleText.length} chars` : 'Unknown error'))
      : null,
  };
}

// ── S7. generateSubjectiveWithGrok (n8n: "Grok - Generate Article (Fallback)") ──

export async function generateSubjectiveWithGrok(systemPrompt: string, userPrompt: string): Promise<string> {
  const resp = await axios.post(
    'https://api.x.ai/v1/chat/completions',
    {
      model: 'grok-4-fast-non-reasoning',
      max_tokens: 5000,
      temperature: 0.4,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user',   content: userPrompt },
      ],
    },
    { headers: { Authorization: `Bearer ${GROK_KEY}`, 'Content-Type': 'application/json' }, timeout: 120_000 },
  );
  return resp.data?.choices?.[0]?.message?.content ?? '';
}

// ── S8. validateSubjective (n8n: "Validate Output - Subjective") ──────────────

export async function validateSubjective(data: SubjectiveGeneratedData): Promise<SubjectiveValidatedData> {
  const articleText = data.articleText;
  const errors: string[] = [];
  const warnings: string[] = [];
  const lowerArticle = articleText.toLowerCase();

  const bannedPhrases = [
    'delve','embark','foster','navigate','harness','unlock','elevate','empower',
    'catalyze','optimize','streamline','tapestry','landscape','journey','blueprint',
    'gateway','realm','catalyst','paradigm shift','pivotal','comprehensive','seamless',
    'vibrant','dynamic','synergistic','multifaceted','robust','transformative','profound',
    'moreover','furthermore','additionally','consequently','conversely','nevertheless',
    'in conclusion','ultimately','to wrap up','in summary','game-changer','ultimate guide',
    'a testament to','it is worth noting','unparalleled','showcase','underscore',
    'highlight','cement','solidify','storied','remarkable','notable','impressive',
    'outstanding','exceptional','incredible','unprecedented','larger than life',
    'household name','the rest is history',
  ];
  const foundBanned = bannedPhrases.filter(p => lowerArticle.includes(p));
  if (foundBanned.length) warnings.push(`Banned phrases: ${foundBanned.join(', ')}`);

  const unsafeWords = [
    'nude','naked','suicide','kill','shot ','stabbed','fake news','conspiracy',
    ' sex ','sexual','sexy','harassment','marijuana','cocaine','assault','scam',
    'drug ','racism','vaccine','rape','molest','damn','porn','smoking','tobacco','wtf',
  ];
  const foundUnsafe = unsafeWords.filter(w => lowerArticle.includes(w.trim().toLowerCase()));
  if (foundUnsafe.length) errors.push(`UNSAFE content: ${foundUnsafe.join(', ')}`);

  const metaMatch = articleText.match(/^META:\s*(.+)$/m);
  if (metaMatch && metaMatch[1].length > 120) warnings.push(`META too long: ${metaMatch[1].length} chars (max 120)`);

  // Em-dash check (titles excluded)
  if (articleText.includes('—')) {
    const slideBlocks = articleText.match(/SLIDE\s*\d+[\s\S]*?(?=SLIDE\s*\d+|SOURCES:|$)/gi) ?? [];
    slideBlocks.forEach((block, i) => {
      const bodyLines = block.split('\n').slice(2).join(' ');
      if (bodyLines.includes('—')) warnings.push(`Slide ${i + 1}: em-dash (banned)`);
    });
  }

  // Parse slides + word counts. Only treat explicit "SLIDE N" markers as slide
  // boundaries — a rank-number title like "21. Game Name" must NOT start a new
  // slide (the old loose regex /^(\d+[.)]|#)/ caused massive double-counting).
  const slideResults: Array<{ slide: number; words: number }> = [];
  let slideNum = 0;
  let slideBody = '';
  let slideTitleSet = false;

  const flushSlide = () => {
    if (slideNum > 0 && slideBody.trim()) {
      const wc = slideBody.trim().split(/\s+/).filter(Boolean).length;
      slideResults.push({ slide: slideNum, words: wc });
      if (slideNum === 1 && wc > 60) warnings.push(`Intro: ${wc} words (max 60)`);
      else if (slideNum > 1 && (wc < 35 || wc > 55)) warnings.push(`Slide ${slideNum}: ${wc} words (expected 35-50)`);
    }
    slideBody = '';
    slideTitleSet = false;
  };

  for (const rawLine of articleText.split('\n')) {
    const line = rawLine.trim().replace(/\*\*/g, '').trim();
    const m = line.match(/^SLIDE\s*(\d+)/i);
    if (m) {
      flushSlide();
      slideNum = parseInt(m[1], 10);
    } else if (slideNum > 0 && line && !line.startsWith('SOURCES') && !line.startsWith('META:')) {
      if (!slideTitleSet) {
        // first non-empty line after the SLIDE marker is the title
        slideTitleSet = true;
      } else {
        slideBody += (slideBody ? ' ' : '') + line;
      }
    }
  }
  flushSlide();

  const expectedTotal = data.slideCount + 1;          // intro + N content slides
  const detectedCount = slideResults.length;
  if (detectedCount < expectedTotal) warnings.push(`Expected ${expectedTotal} slides, detected ${detectedCount}`);

  // Repetitive openings
  const slideBlocks = articleText.match(/SLIDE\s*\d+[\s\S]*?(?=SLIDE\s*\d+|SOURCES:|$)/gi) ?? [];
  const openingWords: string[] = [];
  slideBlocks.forEach((block, i) => {
    if (i === 0) return;
    const bodyLines = block.split('\n').filter(l => l.trim()).slice(1);
    if (bodyLines.length > 0) {
      const firstWord = bodyLines.join(' ').trim().split(/\s+/)[0]?.toLowerCase();
      if (firstWord) openingWords.push(firstWord);
    }
  });
  const wordCounts: Record<string, number> = {};
  openingWords.forEach(w => wordCounts[w] = (wordCounts[w] ?? 0) + 1);
  const overused = Object.entries(wordCounts).filter(([, c]) => c > 2).map(([w, c]) => `"${w}" (${c}x)`);
  if (overused.length) warnings.push(`Repetitive openings: ${overused.join(', ')}`);

  // Must-include coverage
  if (data.mustIncludeItems.length > 0) {
    const missed = data.mustIncludeItems.filter(item => {
      const lower = item.toLowerCase().trim();
      if (lower.startsWith('-') || /\b(include|use|add)\b/.test(lower)) return false;
      return !lowerArticle.includes(lower);
    });
    if (missed.length) warnings.push(`Must-include items potentially missing: ${missed.join(', ')}`);
  }

  // Quality degradation
  const contentSlides = slideResults.filter(s => s.slide > 1);
  if (contentSlides.length >= 10) {
    const thirdSize = Math.ceil(contentSlides.length / 3);
    const avgFirst = contentSlides.slice(0, thirdSize).reduce((s, x) => s + x.words, 0) / thirdSize;
    const avgLast  = contentSlides.slice(-thirdSize).reduce((s, x) => s + x.words, 0) / thirdSize;
    if (avgLast < avgFirst * 0.8) warnings.push(`Quality degradation: last third avg ${Math.round(avgLast)}w vs first third ${Math.round(avgFirst)}w`);
  }

  const validationStatus: SubjectiveValidatedData['validationStatus'] =
    errors.length > 0 ? 'FAILED' : warnings.length > 0 ? 'WARNINGS' : 'PASSED';

  return { ...data, validationStatus, errors, warnings, slideResults };
}

// ── S9. grokSubjectiveStyleAudit (n8n: "Grok - Style Auditor") ────────────────

export async function grokSubjectiveStyleAudit(data: SubjectiveValidatedData): Promise<string> {
  const systemContent = `You are an MSN Slideshow Style, Quality, and Compliance Auditor. This is a SUBJECTIVE article (quotes, opinions, nostalgic lists, personality-driven). Your mandate is writing quality and rule compliance — NOT fact-checking subjective claims.

CRITICAL: YOU DO NOT REWRITE THE ARTICLE
You emit SURGICAL PATCHES. Each patch is a verbatim FIND string from the original and a REPLACE string. Patches are applied to the original article by my pipeline — you must NEVER reproduce the article in your output. Doing so will corrupt the pipeline.

═══════════════════════════════════════════════════════════════
WHAT TO PATCH (in priority order)
═══════════════════════════════════════════════════════════════

1. PUNCTUATION BANS in slide bodies (em-dashes, semicolons, ellipsis):
   FIND a span containing the violation, REPLACE with the same text minus the violation.
   Example: FIND "the moment—a turning point" REPLACE "the moment, a turning point"

2. BANNED AI PHRASES (showcase, highlight, cement, solidify, storied, remarkable, notable, impressive, outstanding, exceptional, incredible, unparalleled, unprecedented, larger than life, household name, the rest is history, delve, embark, foster, navigate, harness, unlock, elevate, empower, catalyze, tapestry, landscape, journey, blueprint, pivotal, vibrant, dynamic, profound, testament, moreover, furthermore, additionally, consequently, in conclusion, ultimately, game-changer):
   Replace with a neutral synonym that fits the sentence. Patch the smallest span needed.

3. BANNED CONTENT WORDS (nude, naked, suicide, kill, sex, sexual, drug, racism, rape, porn, etc.):
   Patch the span to soften or remove the unsafe term while keeping the sentence intact.

4. META TOO LONG (>120 chars): emit a patch with the full META line as FIND and a ≤120-char rewrite as REPLACE.

5. SLIDE WORD COUNT VIOLATIONS (intro >60 words, body slide outside 35-50): only patch if you can tighten without losing meaning. Use the full slide body as FIND.

6. FABRICATED SPECIFICS (a year/event/quote-origin that has no support in source data): patch the specific span to soften or remove the invented precision. Keep the rest of the slide.

7. UNVERIFIABLE QUOTE ORIGIN CLAIMS ("from his 2009 USC speech" when no source confirms): patch the origin clause out, keep the quote.

═══════════════════════════════════════════════════════════════
WHAT NOT TO PATCH
═══════════════════════════════════════════════════════════════

- DO NOT flag opinions ("best ever", "greatest"). Subjective claims are allowed.
- DO NOT touch widely attributed quotes — those are acceptable as-is.
- DO NOT reorder slides.
- DO NOT add new slides (use MISSING_ITEM block instead, see below).
- DO NOT touch slide titles unless they contain a banned phrase or content word.
- DO NOT emit a patch where FIND == REPLACE.

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT — FOLLOW EXACTLY
═══════════════════════════════════════════════════════════════

=== PATCHES ===

<<<PATCH>>>
SCOPE: SLIDE 5
FIND:
the exact text from the original article — verbatim, with all original whitespace and punctuation
END_FIND
REPLACE:
the corrected text
END_REPLACE
REASON: banned phrase "showcase"
<<<END>>>

<<<PATCH>>>
SCOPE: META
FIND:
showcase—ranked moment
END_FIND
REPLACE:
ranked moment
END_REPLACE
REASON: em-dash + banned phrase
<<<END>>>

(emit one block per fix. No blocks if nothing needs fixing.)

=== END PATCHES ===

=== MISSING ITEMS ===
(One line per mandatory item not present in the article. Format: ITEM: "name". If all mandatory items are present, write: ALL PRESENT.)
=== END MISSING ===

=== SUMMARY ===
One sentence on what was checked and how many patches were emitted.
=== END SUMMARY ===

═══════════════════════════════════════════════════════════════
HARD RULES FOR PATCHES
═══════════════════════════════════════════════════════════════

- The FIND text MUST appear verbatim in the original article — copy-paste exactly. If you cannot copy verbatim, do not emit the patch.
- Keep FIND short — the smallest span that contains the violation (ideally one phrase, max one sentence).
- REPLACE must be a clean rewrite of FIND only — do not invent new facts.
- One patch per violation. Do not bundle.
- ABSOLUTELY DO NOT output the article, or any slide body, anywhere outside the PATCHES blocks.`;

  const userContent = `AUDIT THIS MSN SUBJECTIVE SLIDESHOW. Emit surgical patches only.\n\nTitle: "${data.title}"\nCategory: ${data.category}\nArticle Type: ${data.articleType}\nTone Dial: ${data.toneDial}\nPrimary Source URL: ${data.primarySourceUrl || 'None'}\n\nARTICLE TO AUDIT:\n${data.articleText}\n\nMANDATORY ITEMS THAT MUST APPEAR IN THE ARTICLE:\n${data.hasMustInclude ? data.mustIncludeItems.map((item, i) => (i + 1) + '. ' + item).join('\n') : 'None specified.'}\n\nOutput patches in the exact PATCHES format. Never reproduce the article.`;

  try {
    const resp = await axios.post(
      'https://api.x.ai/v1/chat/completions',
      {
        model: 'grok-4-fast-non-reasoning',
        max_tokens: 8000,
        temperature: 0.0,
        messages: [
          { role: 'system', content: systemContent },
          { role: 'user',   content: userContent },
        ],
      },
      { headers: { Authorization: `Bearer ${GROK_KEY}`, 'Content-Type': 'application/json' }, timeout: 240_000 },
    );
    return resp.data?.choices?.[0]?.message?.content ?? '';
  } catch (err: unknown) {
    if (axios.isAxiosError(err)) {
      console.error(`[grokSubjectiveStyleAudit] HTTP ${err.response?.status}: ${JSON.stringify(err.response?.data ?? err.message)}`);
    }
    return '';
  }
}

// ── S10. extractSubjectiveAudit (n8n: "Extract Audited Article - Subjective") ──

export async function extractSubjectiveAudit(data: SubjectiveValidatedData, grokText: string): Promise<SubjectiveAuditedData> {
  const originalArticleText = data.articleText;

  // Empty/missing audit — keep original, mark not audited.
  if (!grokText || grokText.trim().length < 50) {
    return {
      ...data,
      originalArticleText,
      auditedArticle: originalArticleText,
      auditReport: '=== AUDIT REPORT ===\nAudit unavailable — original article kept unchanged.\n=== END AUDIT REPORT ===',
      summaryComment: 'Grok audit failed — original article used unchanged.',
      articleText: originalArticleText,
      wasAudited: false,
    };
  }

  // 1) Parse surgical patches and apply them to the original article.
  const patches = parseAuditPatches(grokText);
  const { result: auditedArticle, applied, skipped, log } = applyAuditPatches(originalArticleText, patches);

  // 2) Extract MISSING ITEMS list (mandatory items the auditor couldn't fix
  //    with a patch — these get flagged to the writer, not auto-inserted).
  const missingMatch = grokText.match(/=== MISSING ITEMS ===\s*([\s\S]*?)\s*=== END MISSING ===/i);
  const missingBlock = missingMatch?.[1]?.trim() ?? '';
  const missingItems = (missingBlock && !/^ALL PRESENT$/i.test(missingBlock))
    ? [...missingBlock.matchAll(/ITEM:\s*"([^"]+)"/gi)].map(m => m[1])
    : [];

  // 3) Extract human summary block.
  const summaryMatch = grokText.match(/=== SUMMARY ===\s*([\s\S]*?)\s*=== END SUMMARY ===/i);
  const summaryFromGrok = summaryMatch?.[1]?.trim() ?? '';

  // 4) Compose an audit report we can show in the final output.
  const auditReport = [
    '=== AUDIT REPORT ===',
    `Patches emitted: ${patches.length}`,
    `Patches applied: ${applied}`,
    `Patches skipped (FIND not located): ${skipped}`,
    log.length > 0 ? '\nPatch log:\n  ' + log.join('\n  ') : '',
    missingItems.length > 0 ? `\nMandatory items missing (writer review): ${missingItems.join(', ')}` : '',
    '=== END AUDIT REPORT ===',
  ].filter(Boolean).join('\n');

  const summaryComment = summaryFromGrok
    || (applied > 0
      ? `Applied ${applied} surgical patch${applied === 1 ? '' : 'es'}${skipped > 0 ? ` (${skipped} skipped)` : ''}${missingItems.length > 0 ? `; ${missingItems.length} mandatory item(s) flagged` : ''}.`
      : 'No patches needed — article passed style audit.');

  console.log(`[extractSubjectiveAudit] applied=${applied} skipped=${skipped} missing=${missingItems.length}`);

  // Safety: surgical patches can never lose slides, but verify anyway.
  const origSlides = (originalArticleText.match(/SLIDE\s*\d+/gi) ?? []).length;
  const newSlides  = (auditedArticle.match(/SLIDE\s*\d+/gi) ?? []).length;
  const finalArticle = newSlides >= origSlides ? auditedArticle : originalArticleText;
  if (newSlides < origSlides) {
    console.warn(`[extractSubjectiveAudit] Patch application lost slides (${origSlides} → ${newSlides}). Reverted to original.`);
  }

  return {
    ...data,
    originalArticleText,
    auditedArticle: finalArticle,
    auditReport,
    summaryComment,
    articleText: finalArticle,
    wasAudited: applied > 0 || patches.length === 0,
  };
}

// ── S11. finalAssemblySubjective ──────────────────────────────────────────────

export async function finalAssemblySubjective(data: SubjectiveAuditedData): Promise<FinalOutput> {
  // Subjective quality: research 25% + structural 35% + style audit 40%.
  // No fact-verification score (we don't fact-check subjective claims).
  const researchScore   = data.researchOk ? 80 : 50;
  const structuralScore = data.validationStatus === 'PASSED' ? 100 : data.validationStatus === 'WARNINGS' ? 70 : 40;
  const styleScore      = data.wasAudited && data.auditedArticle.length > 200 ? 90 : 60;
  const qualityScore    = Math.round((researchScore * 0.25) + (structuralScore * 0.35) + (styleScore * 0.40));

  const summaryParts: string[] = [];
  if (data.errors.length)   summaryParts.push(`Errors: ${data.errors.join('; ')}`);
  if (data.warnings.length) summaryParts.push(`Warnings: ${data.warnings.length}`);
  if (data.summaryComment)  summaryParts.push(data.summaryComment);
  if (data.generatedBy)     summaryParts.push(`Generated by: ${data.generatedBy}`);

  const auditReport = `
QUALITY AUDIT REPORT (Subjective)

OVERALL QUALITY SCORE: ${qualityScore}/100

RESEARCH: ${researchScore}/100
STRUCTURE: ${structuralScore}/100
STYLE AUDIT: ${styleScore}/100

ARTICLE TYPE: ${data.articleType}
TONE DIAL: ${data.toneDial}

VALIDATION
Status: ${data.validationStatus}
Errors: ${data.errors.join(', ') || 'None'}
Warnings: ${data.warnings.join(', ') || 'None'}

STYLE AUDIT (Grok)
${data.auditReport}
`;

  return {
    title: data.title, category: data.category, slideCount: data.slideCount, writerName: data.writerName,
    articleText:         data.articleText,
    originalArticleText: data.originalArticleText || data.articleText,
    auditReport, qualityScore, researchScore,
    verificationScore: styleScore, structuralScore, originalityScore: styleScore,
    primarySourceUrl:       data.primarySourceUrl,
    combinedSourceListText: data.sourceList,
    summaryComment: summaryParts.length > 0 ? summaryParts.join('. ') : 'Subjective article passed all checks.',
    validationStatus: data.validationStatus,
    factsVerified:    'N/A (subjective)',
    grokRulesPassed:  data.wasAudited ? 'Audited' : 'Skipped',
    flagsForReview:   data.errors.length > 0 ? data.errors.join('; ') : 'None',
    generatedBy:      data.generatedBy || 'Unknown',
    generatedAt:      new Date().toISOString(),
  };
}

