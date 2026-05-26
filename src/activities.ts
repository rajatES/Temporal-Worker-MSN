import axios from 'axios';
import * as dotenv from 'dotenv';
import {
  FormInput, PreparedData, SourcedData, AtomizedData, ResearchedData,
  MergedData, PromptData, GeneratedData, ValidatedData, ClaimedData,
  VerifiedData, AuditedData, FinalOutput, TemporalCtx,
  FormatConfig, TitleAnalysis, AtomizedFact, SourceEntry, FactProvenance,
  SubjectivePreparedData, SubjectiveSourcedData, SubjectiveAtomizedData,
  SubjectiveResearchedData, SubjectiveMergedData,
  SubjectivePromptData, SubjectiveGeneratedData, SubjectiveValidatedData,
  SubjectiveAuditedData,
} from './types';

dotenv.config();

const FIRECRAWL_KEY  = process.env.FIRECRAWL_API_KEY!;
const PERPLEXITY_KEY = process.env.PERPLEXITY_API_KEY!;
const ANTHROPIC_KEY  = process.env.ANTHROPIC_API_KEY!;
const GROK_KEY       = process.env.GROK_API_KEY!;
const VERCEL_GATEWAY_KEY = process.env.VERCEL_AI_GATEWAY_KEY!;

const VERCEL_GATEWAY_URL = 'https://ai-gateway.vercel.sh/v1/responses';
const XAI_DIRECT_URL     = 'https://api.x.ai/v1/chat/completions';

function extractGrokResponseText(data: unknown): string {
  const d = data as Record<string, unknown>;
  const outputs = (d?.output as Array<Record<string, unknown>>) ?? [];
  if (outputs.length > 0) {
    const msgOutput = [...outputs].reverse().find((o: Record<string, unknown>) => o.type === 'message');
    if (msgOutput?.content) {
      const contentArr = msgOutput.content as Array<{ type: string; text?: string }>;
      const text = contentArr.find(c => c.type === 'output_text')?.text;
      if (text) return text;
    }
  }
  const fallback = d as { choices?: Array<{ message: { content: string } }> };
  return fallback?.choices?.[0]?.message?.content ?? '';
}

// ── Grok helper: Vercel AI Gateway → direct xAI API fallback ────────────────
// Tries the Vercel gateway first (Responses API format). If it returns a
// retryable error (403 Forbidden, 429 Rate-limit, or 5xx), falls back to the
// direct xAI Chat Completions API using GROK_API_KEY. Returns the raw response
// data — callers use extractGrokResponseText() on top as needed.

async function callGrokWithFallback(opts: {
  systemContent: string;
  userContent: string;
  maxTokens: number;
  temperature: number;
  timeout: number;
  tools?: Array<{ type: string }>;
  label: string;
}): Promise<unknown> {
  // ── 1. Try Vercel AI Gateway (Responses API format) ──────────────────────
  try {
    const body: Record<string, unknown> = {
      model: 'xai/grok-4.3',
      max_output_tokens: opts.maxTokens,
      temperature: opts.temperature,
      input: [
        { role: 'system', content: opts.systemContent },
        { role: 'user',   content: opts.userContent },
      ],
    };
    if (opts.tools) body.tools = opts.tools;

    const resp = await axios.post(VERCEL_GATEWAY_URL, body, {
      headers: { Authorization: `Bearer ${VERCEL_GATEWAY_KEY}`, 'Content-Type': 'application/json' },
      timeout: opts.timeout,
    });
    return resp.data;
  } catch (err: unknown) {
    const status = axios.isAxiosError(err) ? err.response?.status : undefined;
    const detail = axios.isAxiosError(err)
      ? JSON.stringify(err.response?.data ?? err.message)
      : String(err);
    const retryable = status && (status === 403 || status === 429 || status >= 500);
    console.warn(
      `[${opts.label}] Vercel gateway failed (HTTP ${status}): ${detail}.` +
      (retryable ? ' Falling back to direct xAI API…' : ' NOT retryable — re-throwing.'),
    );
    if (!retryable) throw err;
  }

  // ── 2. Fallback: direct xAI Chat Completions API ────────────────────────
  if (!GROK_KEY) {
    throw new Error(
      `[${opts.label}] Vercel gateway returned 403/429/5xx and GROK_API_KEY is not set — cannot fall back to direct xAI API`,
    );
  }

  console.log(`[${opts.label}] Attempting direct xAI API (${XAI_DIRECT_URL})…`);
  const body: Record<string, unknown> = {
    model: 'grok-4.3',
    max_tokens: opts.maxTokens,
    temperature: opts.temperature,
    messages: [
      { role: 'system', content: opts.systemContent },
      { role: 'user',   content: opts.userContent },
    ],
  };
  // Note: tools (web_search) are Vercel-gateway-specific; the direct API
  // does not support the same format, so Grok falls back to training data.

  const resp = await axios.post(XAI_DIRECT_URL, body, {
    headers: { Authorization: `Bearer ${GROK_KEY}`, 'Content-Type': 'application/json' },
    timeout: opts.timeout,
  });
  console.log(`[${opts.label}] Direct xAI API succeeded.`);
  return resp.data;
}

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

// ─────────────────────────────────────────────────────────────────────────────
// parseWordCountOverride — detect explicit word-count instructions in userContext
// e.g. "40-45 words only", "max 40 words", "keep it under 45 words"
// ─────────────────────────────────────────────────────────────────────────────

function parseWordCountOverride(userContext: string): { min: number; max: number } | undefined {
  if (!userContext) return undefined;
  const lower = userContext.toLowerCase();
  let m: RegExpMatchArray | null;
  // "40-45 words" or "40 to 45 words"
  m = lower.match(/(\d{2,3})\s*(?:[-–—]|to)\s*(\d{2,3})\s*words/);
  if (m && m[1] && m[2]) return { min: parseInt(m[1]), max: parseInt(m[2]) };
  // "max 45 words" / "under 45 words" / "no more than 45 words"
  m = lower.match(/(?:max|maximum|under|no more than|at most|limit)\s*(\d{2,3})\s*words/);
  if (m) return { min: Math.max(20, parseInt(m[1]) - 10), max: parseInt(m[1]) };
  // "45 words max" / "45 words only" / "45 words or less"
  m = lower.match(/(\d{2,3})\s*words?\s*(?:max|maximum|limit|only|or less)/);
  if (m) return { min: Math.max(20, parseInt(m[1]) - 10), max: parseInt(m[1]) };
  // "keep it to 40 words" / "around 40 words"
  m = lower.match(/(?:keep|aim|target|around|approximately|about)\s*(?:it\s*)?(?:to\s*)?(\d{2,3})\s*words/);
  if (m) { const n = parseInt(m[1]); return { min: Math.max(20, n - 5), max: n + 5 }; }
  return undefined;
}

// parseMustIncludeItems — robust extraction of must-include items from any
// user-supplied format: markdown tables, numbered/bulleted lists, plain lines,
// or mixed. Strips source rank numbers, table formatting, header/separator
// rows, and deduplicates by normalised entity name.
// ─────────────────────────────────────────────────────────────────────────────

function parseMustIncludeItems(raw: string): string[] {
  if (!raw || !raw.trim()) return [];

  const lines = raw.split(/\r?\n/).map(l => l.trim()).filter(Boolean);

  // Detect if this is a markdown/pipe table: 2+ lines contain at least 2 pipes
  const pipeLines = lines.filter(l => (l.match(/\|/g) ?? []).length >= 2);
  const isTable = pipeLines.length >= 2;

  // Detect tab-separated table: 2+ lines contain 2+ tab characters
  const tabLines = lines.filter(l => (l.match(/\t/g) ?? []).length >= 2);
  const isTsv = !isTable && tabLines.length >= 2;

  const extracted: string[] = [];

  if (isTable) {
    for (const line of lines) {
      // Skip separator rows: | -- | --- | etc.
      if (/^\|[\s\-:|]+\|$/.test(line)) continue;

      // Split pipe-delimited cells
      const cells = line.split('|').map(c => c.trim()).filter(Boolean);
      if (cells.length < 2) continue;

      // Skip header rows: detect by checking if first cell is a common header
      // keyword or if all cells look like column headers (no digits in any cell)
      const firstLower = cells[0].toLowerCase();
      if (/^(#|rank|no\.?|number|pos\.?|position|sr\.?)$/i.test(firstLower)) continue;
      if (cells.every(c => /^[a-z\s.()\/&]+$/i.test(c) && !/\d{4}/.test(c) && c.length < 30)) continue;

      // Try to identify: rank-number cell, name cell(s), and optional context cells
      // The first cell that's purely numeric (or #N) is the source rank — skip it
      let nameParts: string[] = [];
      let contextParts: string[] = [];
      let seenName = false;

      for (const cell of cells) {
        // Pure rank number — skip
        if (/^\d{1,3}$/.test(cell) || /^#\d+$/.test(cell)) continue;

        // Year-only cell (e.g. "1993", "1985–1990") → context
        if (/^\d{4}(\s*[–\-]\s*\d{4})?$/.test(cell)) {
          contextParts.push(cell);
          continue;
        }

        if (!seenName) {
          nameParts.push(cell);
          seenName = true;
        } else {
          contextParts.push(cell);
        }
      }

      if (nameParts.length === 0) continue;

      // Build a clean item string: "Name — Context1, Context2" or just "Name"
      const name = nameParts.join(' ').trim();
      const context = contextParts.filter(c => c.length > 0).join(', ').trim();
      const item = context ? `${name} — ${context}` : name;
      if (item.length > 2) extracted.push(item);
    }
  } else if (isTsv) {
    for (const line of lines) {
      if (/^[\s\-=:]+$/.test(line.replace(/\t/g, ''))) continue;
      const cols = line.split('\t').map(c => c.trim()).filter(Boolean);
      if (cols.length < 2) continue;
      if (/^(#|rank|no\.?|number|pos\.?|position|sr\.?|team|name|player)$/i.test(cols[0])) continue;
      if (cols.every(c => /^[a-z\s.()\/&]+$/i.test(c) && !/\d{4}/.test(c) && c.length < 30)) continue;
      let nameCol = cols[0];
      if (/^\d{1,3}$/.test(nameCol) || /^#\d+$/.test(nameCol)) {
        nameCol = cols[1] ?? '';
      }
      const name = nameCol.trim();
      if (name.length > 2) extracted.push(name);
    }
  } else {
    // Non-table: numbered list, bulleted list, comma-separated, or plain lines.
    // Handles any mix: "1. A, 2. B", "A\nB\nC", "1. A 2. B 3. C" (no delimiter),
    // or single-word items like "Sektori, Dispatch, BALL x PIT".
    for (const line of lines) {
      // Strip leading list marker: "1.", "1)", "#1", "- ", "* ", "•"
      let cleaned = line
        .replace(/^(\d{1,3}[.)]\s*|#\d+[.):]?\s*|[-*•]\s+)/, '')
        .trim();

      if (!cleaned) continue;

      // ── Inline-numbered items on one line ──────────────────────────────
      // Catches "Game A 2. Game B 3. Game C" or "Game A, 2. Game B, 3. Game C"
      // after the leading "1." was already stripped above.
      const numberedParts = cleaned.split(/,?\s+(?=\d{1,3}[.)]\s|#\d+\s)/);
      if (numberedParts.length >= 2) {
        for (const part of numberedParts) {
          const stripped = part
            .replace(/^\d{1,3}[.)]\s*/, '')
            .replace(/^#\d+[.):]?\s*/, '')
            .replace(/,\s*$/, '')
            .trim();
          if (stripped.length > 2) extracted.push(stripLeadingRank(stripped));
        }
        continue;
      }

      // ── Comma-separated items ──────────────────────────────────────────
      // Split on commas (but not commas inside parentheses).
      // 3+ parts = always a list. 2 parts = only split if both have 2+ words
      // (avoids false splits like "Game Title, 2025").
      const commaParts = cleaned.split(/,(?!\d{3}(?:\D|$))(?![^(]*\))/).map(s => s.trim()).filter(Boolean);
      if (commaParts.length >= 3 ||
          (commaParts.length === 2 && commaParts.every(p => p.split(/\s+/).length >= 2))) {
        for (const part of commaParts) {
          if (part.length > 2) extracted.push(stripLeadingRank(part));
        }
        continue;
      }

      // ── Single item ────────────────────────────────────────────────────
      if (cleaned.length > 2) extracted.push(stripLeadingRank(cleaned));
    }
  }

  // Deduplicate by normalised form (lowercase, stripped of parenthetical aliases)
  const seen = new Set<string>();
  const deduped: string[] = [];
  for (const item of extracted) {
    const norm = item.toLowerCase().replace(/\s*\(.*?\)\s*/g, ' ').replace(/\s+/g, ' ').trim();
    if (!seen.has(norm)) {
      seen.add(norm);
      deduped.push(item);
    }
  }

  return deduped;
}

// Strip a leading rank number that may survive from non-table formats
// e.g. "40 Dr. Jack Griffin — The Invisible Man" → "Dr. Jack Griffin — The Invisible Man"
// but preserve names that start with a number: "50 Cent", "21 Savage"
function stripLeadingRank(s: string): string {
  // Match: digits at start, followed by a period/colon/dash/whitespace, then a
  // word that starts with uppercase (indicating a name follows the rank).
  // Don't strip if the digits are part of the name (no separator between rank and name).
  const m = s.match(/^(\d{1,3})\s*[.):\-–—]?\s+([A-Z])/);
  if (m) {
    return s.slice(m.index! + m[0].length - 1).trim(); // start at the uppercase letter
  }
  return s;
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
  const isRanking    = /\b(top|best|greatest|worst|most|ranked|ranking|highest|lowest)\b/i.test(title);
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
    .filter((url, i, arr) => arr.findIndex(u => u.replace(/\/+$/, '') === url.replace(/\/+$/, '')) === i)
    .slice(0, 5);

  const userPrimaryUrl      = allUrls[0] ?? '';
  const userSecondaryUrls   = allUrls.slice(1);                // up to 4 extras (5 total)
  const isUserUrlRestricted = userPrimaryUrl ? isRestricted(userPrimaryUrl) : false;
  const hasValidUserSource  = !!userPrimaryUrl && !isUserUrlRestricted;
  const mustIncludeItems    = parseMustIncludeItems(mustIncludeRaw ?? '');
  const userWordCountOverride = parseWordCountOverride(userContext);

  return {
    title, category, slideCount, writerName, userContext, writingStyle,
    userPrimaryUrl, userSecondaryUrls, hasValidUserSource, isUserUrlRestricted,
    restrictedDomains: RESTRICTED_DOMAINS, mustIncludeItems,
    hasMustInclude: mustIncludeItems.length > 0,
    temporalContext, formatConfig, titleAnalysis,
    sourceCount: allUrls.length,
    timestamp: new Date().toISOString(),
    isSports: category.startsWith('Sports'),
    userWordCountOverride,
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
        throw new Error(`Firecrawl returned empty markdown for ${url}`);
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
      throw new Error(`Firecrawl scrape failed for ${url}: ${errBody}`);
    }
  }
  throw new Error(`Firecrawl scrape failed after all attempts for ${url}`);
}

// ── 3. analyzeSourceAlignment (n8n: "Analyze Source Alignment") ──────────────

export async function analyzeSourceAlignment(
  prepData: PreparedData,
  primaryMarkdown: string,
  additionalMarkdowns: string[] = [],
): Promise<SourcedData> {
  // ── ALL SOURCES TREATED EQUALLY ─────────────────────────────────────────
  // No primary/secondary/tertiary tiering — each scraped URL is an equally
  // authoritative Tier 1A source. atomizeFacts uses `scrapedSources` to
  // atomize each independently and merge facts by item name.
  const scrapedSources: Array<{ url: string; markdown: string }> = [];
  if (primaryMarkdown) scrapedSources.push({ url: prepData.userPrimaryUrl, markdown: primaryMarkdown });
  additionalMarkdowns.forEach((md, i) => {
    if (md && md.length > 0) {
      scrapedSources.push({ url: prepData.userSecondaryUrls[i] ?? '', markdown: md });
    }
  });

  const sourceSections: string[] = scrapedSources.map(s => `=== SOURCE: ${s.url} ===\n${s.markdown}`);
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
      scrapedSources,
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

/**
 * Merge atomized items that refer to the same entity across different sources.
 * Match key = sorted lowercase tokens of itemName (length > 2). So
 * "Caleb Williams — Chicago Bears" and "Chicago Bears: Caleb Williams" merge
 * because they share the same tokens. Different specificity (just "Caleb
 * Williams" vs the full form) stays separate — by design.
 *
 * All scraped sources are EQUAL Tier 1A authority. There is no priority
 * tiering — facts from any source are unioned (deduped by type+value).
 */
function mergeAtomizedItems(items: AtomizedFact[]): AtomizedFact[] {
  const normalizeKey = (name: string): string => {
    if (!name) return '';
    return name.toLowerCase()
      .replace(/[—–:,()]/g, ' ')
      .replace(/\s+/g, ' ')
      .trim()
      .split(' ')
      .filter(t => t.length > 2)
      .sort()
      .join(' ');
  };
  const factKey = (f: { type: string; value: string }) => `${f.type}|${f.value.toLowerCase()}`;
  const byKey = new Map<string, AtomizedFact>();
  const orderKeys: string[] = [];

  for (const item of items) {
    const key = normalizeKey(item.itemName);
    const existing = byKey.get(key);
    if (!existing) {
      byKey.set(key, { ...item, facts: [...item.facts] });
      orderKeys.push(key);
      continue;
    }
    // Union facts (dedupe)
    const seen = new Set(existing.facts.map(factKey));
    for (const f of item.facts) {
      if (!seen.has(factKey(f))) { existing.facts.push(f); seen.add(factKey(f)); }
    }
    // Concatenate rawContent (cap)
    if (existing.rawContent.length < 600 && item.rawContent) {
      existing.rawContent = (existing.rawContent + '\n' + item.rawContent).slice(0, 600);
    }
    // Concatenate narrativeContext (cap)
    if (item.narrativeContext) {
      if (!existing.narrativeContext) {
        existing.narrativeContext = item.narrativeContext;
      } else if (existing.narrativeContext.length < 500) {
        existing.narrativeContext = (existing.narrativeContext + ' ' + item.narrativeContext).slice(0, 600);
      }
    }
  }
  const merged = orderKeys.map(k => byKey.get(k)!);
  // Multi-source items can accumulate many facts — bump cap to 24
  for (const item of merged) item.facts = item.facts.slice(0, 24);
  return merged;
}

export async function atomizeFacts(data: SourcedData): Promise<AtomizedData> {
  // Multi-source equality: each scraped URL is an equally authoritative Tier 1A
  // source. We atomize each independently, then merge facts by item name so
  // duplicates across sources combine instead of bloating the prompt.
  const scrapedSources = data.sourceAnalysis?.scrapedSources ?? [];
  const sourcesToAtomize: string[] = scrapedSources.length > 0
    ? scrapedSources.map(s => s.markdown).filter(md => md && md.length > 0)
    : (data.sourceAnalysis?.scrapedContent ? [data.sourceAnalysis.scrapedContent] : []);

  const combinedRawContent = [sourcesToAtomize.join('\n\n'), data.userContext].filter(Boolean).join('\n\n');

  if (!combinedRawContent || combinedRawContent.length < 50) {
    return { ...data, atomizedFacts: [], factOnlyRepresentation: '', sourceSignatures: [], atomizationStats: { itemsProcessed: 0, totalFacts: 0 } };
  }

  const rawItems: AtomizedFact[] = [];
  const sourceSignatures: string[]    = [];

  // Atomize each source independently — equal Tier 1A authority across all.
  for (const sourceContent of sourcesToAtomize) {
    if (!sourceContent || sourceContent.length < 50) continue;

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
    // We consume the leading newline as part of the delimiter so each section
    // starts cleanly with `##` — without this, inner regexes anchored at `^##`
    // fail because sections begin with `\n`. The regex also requires a literal
    // separator (`.`, `)`, or `:`) between the digit and the title so it doesn't
    // accidentally match content-internal patterns like "selected #1 overall".
    if (sections.length < 3) {
      sections = sourceContent.split(/\n(?=[ \t]*#{1,2}[ \t]+\d{1,3}[.):][ \t]+)/);
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
      // Requires a literal `:` separator — without this it swallows the numbered
      // format "## 1. Player — Team" by treating the whole heading as `team` and
      // the first content line as `subject`.
      const headingMatch = section.match(/^##\s*\[?([^\]\n]+?)\]?(?:\([^)]*\))?\s*:\s+\[?([^\]\n(]+?)\]?(?:\([^)]*\))?\s*(?:\n|$)/);
      if (headingMatch) {
        const team = headingMatch[1].trim();
        const subject = headingMatch[2]?.trim().replace(/\*\*/g, '').replace(/\[([^\]]+)\]\([^)]*\)/g, '$1') ?? '';
        itemName = subject ? `${team}: ${subject}` : team;
        content = section.replace(/^##\s*[^\n]+\n/, '');
      } else {
        // Pattern C: Original numbered format (## 1. Title) — must match the
        // same shape the splitter accepted: line-start, space, separator required.
        const numberedMatch = section.match(/^[ \t]*#{1,2}[ \t]+(\d{1,3})[.):][ \t]+(.+?)(?:\n|$)/);
        if (numberedMatch) {
          itemNumber = parseInt(numberedMatch[1]);
          itemName = numberedMatch[2].trim().replace(/\*\*/g, '');
          content = section.replace(/^[ \t]*#{1,2}[ \t]+\d{1,3}[.):][ \t]+.+?\n/, '');
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

    // ── Non-numeric fact patterns ───────────────────────────────────────────
    // Capture team affiliations, draft picks, transactions, positions, status —
    // the kinds of facts that change recently and don't survive a pure-numeric
    // atomization. These let Claude write accurate, time-sensitive prose
    // (current team, latest draft, recent trade) without relying on training.
    const narrativePatterns: StatDef[] = [
      // Draft picks: numeric ("drafted #1 overall by the Bears", "3rd by Chicago")
      // OR word-ordinal ("selected first overall by the Bears", "picked second")
      { pattern: /\b(?:drafted|selected|picked|chosen)\s+(?:(?:#?\d+(?:st|nd|rd|th)?|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+(?:overall\s+)?)?(?:in\s+(?:the\s+)?\d+(?:st|nd|rd|th)\s+round\s+)?by\s+(?:the\s+)?([A-Z][a-zA-Z0-9 .&'-]{2,40}?)(?=[,.\n]|\s+(?:in|with|after))/gi, type: 'draft' },
      // "picked Drake Maye third overall" — verb + 1-4-word name + ordinal + overall (optionally + "by team")
      { pattern: /\b(?:drafted|selected|picked|chose|chosen)\s+[A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'.-]+){0,3}\s+(?:#?\d+(?:st|nd|rd|th)?|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+overall(?:\s+by\s+(?:the\s+)?[A-Z][a-zA-Z'.& -]{2,30})?\b/gi, type: 'draft' },
      // "Commanders selected him second overall" — team + verb + pronoun + ordinal
      { pattern: /\b[A-Z][a-zA-Z'.-]+(?:\s+[A-Z][a-zA-Z'.-]+){0,2}\s+(?:drafted|selected|picked|chose|chosen)\s+(?:him|her|them)\s+(?:#?\d+(?:st|nd|rd|th)?|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+overall\b/gi, type: 'draft' },
      // "picked X overall" / "selected X overall" (no "by" required) — bare ordinal pick
      { pattern: /\b(?:drafted|selected|picked|chosen)\s+(?:#?\d+(?:st|nd|rd|th)?|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+overall\b/gi, type: 'draft_pick' },
      { pattern: /\b(?:#|No\.?\s*)(\d{1,2})(?:st|nd|rd|th)?\s+(?:overall\s+(?:pick|selection)|pick|selection)\b/gi, type: 'draft_pick' },
      { pattern: /\b\d+(?:st|nd|rd|th)\s+overall\s+(?:pick|selection)\b/gi, type: 'draft_pick' },
      { pattern: /\b(?:going|went)\s+(?:#?\d+(?:st|nd|rd|th)?|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+overall\s+to\s+(?:the\s+)?([A-Z][a-zA-Z0-9 .&'-]{2,40}?)(?=[,.\n])/gi, type: 'draft' },
      // Big contract signings: "signed a four-year, $39.5 million [rookie] contract/deal/extension"
      // Allows up to two intervening adjectives between the dollar amount and the contract noun.
      { pattern: /\bsigned\s+(?:a\s+)?(?:\w+(?:-year)?(?:[,\s]+\w+)?[,\s]+)?\$\d+(?:\.\d+)?\s*(?:million|billion|M|B)?(?:\s+\w+){0,2}\s+(?:contract|deal|extension|agreement)/gi, type: 'contract' },
      // Transactions: trades, signings, free-agent moves
      { pattern: /\b(?:traded|sent|dealt|moved|transferred)\s+to\s+(?:the\s+)?([A-Z][a-zA-Z0-9 .&'-]{2,40}?)(?=[,.\n]|\s+(?:in|for|with))/g, type: 'transaction' },
      { pattern: /\b(?:signed|inked|agreed)\s+(?:a\s+(?:contract|deal|extension)\s+)?with\s+(?:the\s+)?([A-Z][a-zA-Z0-9 .&'-]{2,40}?)(?=[,.\n])/g, type: 'transaction' },
      { pattern: /\b(?:joined|joins|joining)\s+(?:the\s+)?([A-Z][a-zA-Z0-9 .&'-]{2,40}?)(?=[,.\n]|\s+(?:in|as|after))/g, type: 'transaction' },
      // Current team affiliation
      { pattern: /\b(?:plays?|currently plays|now plays|started|stars?)\s+for\s+(?:the\s+)?([A-Z][a-zA-Z0-9 .&'-]{2,40}?)(?=[,.\n])/g, type: 'affiliation' },
      { pattern: /\bwith\s+(?:the\s+)?([A-Z][a-zA-Z][a-zA-Z0-9 .&'-]{3,40}?)(?=[,.\n]|\s+(?:since|after|in\s+\d))/g, type: 'affiliation' },
      // Positions
      { pattern: /\b(?:starting|veteran|rookie|backup|All-Pro|Pro Bowl)\s+(quarterback|QB|running back|RB|wide receiver|WR|tight end|TE|cornerback|CB|safety|linebacker|LB|defensive end|edge rusher|defensive tackle|center|guard|tackle|kicker|punter|point guard|shooting guard|small forward|power forward|center|striker|midfielder|defender|goalkeeper)\b/gi, type: 'position' },
      // Status changes
      { pattern: /\b(retired|retiring|injured|suspended|released|cut|waived|placed on IR|on injured reserve|inactive|active roster)\b/gi, type: 'status' },
    ];

    for (const { pattern, type } of narrativePatterns) {
      const rx = new RegExp(pattern.source, pattern.flags);
      let m: RegExpExecArray | null;
      while ((m = rx.exec(content)) !== null) {
        const value = m[0].trim().replace(/\s+/g, ' ');
        // Skip generic / overlong matches
        if (value.length < 4 || value.length > 80) continue;
        facts.push({ type, value });
      }
    }

    // ── Smart narrative context: signal-scored selection + adaptive sizing ─────
    // Sentences are scored by non-numeric-fact density (proper nouns, action
    // verbs, time refs, positions) and penalized for noise (number-dominated
    // lines, long quotes, markdown). Budget per item scales inversely with how
    // many structured facts the regexes already captured — sparse items get
    // up to 500 chars of CONTEXT, well-covered items get 200.
    const ACTION_VERB_RE  = /\b(drafted|selected|picked|signed|inked|traded|sent|dealt|moved|joined|joins|retired|retiring|released|cut|waived|debut(?:ed)?|started|leads?|finished|won|broke|set|earned|named|hired|fired|coaches?|manages?|plays?|stars?|takes? over|stepped)\b/i;
    const TIME_REF_RE     = /\b(last (?:season|year|month|week)|this (?:season|year|offseason)|currently|recently|since\s+\d{4}|in\s+(?:19|20)\d{2}|now|after)\b/i;
    const POSITION_RE     = /\b(quarterback|QB|running back|RB|wide receiver|WR|tight end|TE|cornerback|CB|safety|linebacker|LB|defensive end|edge|center|guard|tackle|kicker|punter|point guard|shooting guard|forward|striker|midfielder|defender|goalkeeper|head coach|coach|GM|general manager|owner)\b/i;
    const PROPER_NOUN_RE  = /\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+\b/;
    const NUMBER_HEAVY_RE = /(?:\d[\d,.]*[\s,]+){3,}/;
    const LONG_QUOTE_RE   = /"[^"]{20,}"/;
    const MARKDOWN_RE     = /^(?:##|!\[|\[\[|\*\*)/;

    type ScoredSent = { text: string; originalIdx: number; score: number };
    const rawSentences = content
      .split(/(?<=[.!?])\s+/)
      .map((s, i) => ({ text: s.trim(), originalIdx: i }))
      .filter(s => s.text.length > 25 && s.text.length < 280);

    const scored: ScoredSent[] = rawSentences.map(s => {
      let score = 0;
      if (PROPER_NOUN_RE.test(s.text))  score += 2;
      if (ACTION_VERB_RE.test(s.text))  score += 1;
      if (TIME_REF_RE.test(s.text))     score += 1;
      if (POSITION_RE.test(s.text))     score += 1;
      if (NUMBER_HEAVY_RE.test(s.text)) score -= 1;
      if (LONG_QUOTE_RE.test(s.text))   score -= 2;
      if (MARKDOWN_RE.test(s.text))     score -= 2;
      return { ...s, score };
    });

    // Adaptive char budget: more CONTEXT when structured fact extraction was sparse
    const structuredFactCount = facts.length;
    const charBudget = structuredFactCount >= 5 ? 200
                     : structuredFactCount >= 2 ? 350
                     :                            500;

    // Greedy fill: take highest-scoring sentences until budget exhausted
    const candidates = scored.filter(s => s.score >= 1).sort((a, b) => b.score - a.score);
    const chosen: ScoredSent[] = [];
    let runningLength = 0;
    for (const s of candidates) {
      if (runningLength + s.text.length + 1 > charBudget) continue;
      chosen.push(s);
      runningLength += s.text.length + 1;
      if (chosen.length >= 5) break;
    }

    // Restore original reading order so the context flows naturally for Claude
    const narrativeContext = chosen
      .sort((a, b) => a.originalIdx - b.originalIdx)
      .map(s => s.text)
      .join(' ')
      .slice(0, charBudget);

    const sentences = content.split(/[.!?]+/).filter(s => s.trim().length > 40);
    sentences.forEach(sentence => {
      const words = sentence.trim().split(/\s+/);
      for (let i = 0; i < words.length - 5; i++) {
        const phrase = words.slice(i, i + 6).join(' ').toLowerCase();
        if (!/\d.*\d.*\d/.test(phrase)) sourceSignatures.push(phrase);
      }
    });

    if (facts.length > 0 || content.length > 100) {
      rawItems.push({ itemNumber, itemName, facts: facts.slice(0, 18), rawContent: content.slice(0, 400), narrativeContext });
    }
  });
  }  // ← end for-loop over sourcesToAtomize

  // Merge items that match across sources (same entity, different heading format)
  const atomizedFacts: AtomizedFact[] = mergeAtomizedItems(rawItems);
  console.log(`[atomizeFacts] ${rawItems.length} raw items across ${sourcesToAtomize.length} source(s) → ${atomizedFacts.length} merged items`);

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
    const affiliations = item.facts.filter(f => f.type === 'affiliation');
    const drafts       = item.facts.filter(f => f.type === 'draft' || f.type === 'draft_pick');
    const transactions = item.facts.filter(f => f.type === 'transaction');
    const contracts    = item.facts.filter(f => f.type === 'contract');
    const positions    = item.facts.filter(f => f.type === 'position');
    const statuses     = item.facts.filter(f => f.type === 'status');
    if (stats.length)        lines.push(`  STATS: ${stats.map(s => s.value).join(', ')}`);
    if (achievements.length) lines.push(`  ACHIEVEMENTS: ${achievements.map(a => a.value).join(', ')}`);
    if (dates.length)        lines.push(`  DATES: ${[...new Set(dates.map(d => d.value))].join(', ')}`);
    if (affiliations.length) lines.push(`  AFFILIATION: ${[...new Set(affiliations.map(a => a.value))].join('; ')}`);
    if (drafts.length)       lines.push(`  DRAFT: ${[...new Set(drafts.map(d => d.value))].join('; ')}`);
    if (transactions.length) lines.push(`  TRANSACTIONS: ${[...new Set(transactions.map(t => t.value))].join('; ')}`);
    if (contracts.length)    lines.push(`  CONTRACT: ${[...new Set(contracts.map(c => c.value))].join('; ')}`);
    if (positions.length)    lines.push(`  POSITION: ${[...new Set(positions.map(p => p.value))].join(', ')}`);
    if (statuses.length)     lines.push(`  STATUS: ${[...new Set(statuses.map(s => s.value))].join(', ')}`);
    if (quotes.length)       lines.push(`  QUOTES: ${quotes.map(q => `"${q.value}"`).join('; ')}`);
    if (item.narrativeContext) lines.push(`  CONTEXT (reference only — do NOT mirror phrasing): ${item.narrativeContext}`);
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

// ─────────────────────────────────────────────────────────────────────────────
// Light Perplexity Sanitization
// Strips numbers from sentences that mention a Tier 1A/1B entity when those
// numbers DO NOT match a known Tier 1A/1B fact. Numbers in general-context
// sentences are left alone. Also strips direct quotes >30 chars and source URLs.
// ─────────────────────────────────────────────────────────────────────────────

function lightSanitizePerplexity(
  text: string,
  atomizedFacts: AtomizedFact[],
  userContextContent: string,
): { sanitized: string; stats: { originalLength: number; sanitizedLength: number; statPlaceholders: number; quotesRemoved: number; entitiesTracked: number } } {
  if (!text) return { sanitized: '', stats: { originalLength: 0, sanitizedLength: 0, statPlaceholders: 0, quotesRemoved: 0, entitiesTracked: 0 } };

  // Build entity → known-numbers map from atomized facts
  const entityNumberMap: Record<string, Set<string>> = {};
  const allEntityNames: string[] = [];

  (atomizedFacts || []).forEach(item => {
    if (item.itemName === 'USER_CONTEXT_DATA') return;
    const name = (item.itemName || '').trim();
    if (!name || name === `Item ${item.itemNumber}`) return;

    const knownNumbers = new Set<string>();
    (item.facts || []).forEach(f => {
      if (!f.value) return;
      const nums = String(f.value).match(/\d+(?:,\d{3})*(?:\.\d+)?/g) || [];
      nums.forEach(n => {
        knownNumbers.add(n.replace(/,/g, ''));
        knownNumbers.add(n);
      });
    });

    entityNumberMap[name.toLowerCase()] = knownNumbers;
    allEntityNames.push(name);

    // ── Fuzzy entity indexing across separator-joined item names ──────────
    // itemName may take many forms:
    //   "Caleb Williams"                  (person only)
    //   "Caleb Williams — Chicago Bears"  (person — team, em-dash)
    //   "Chicago Bears: Caleb Williams"   (team: person, colon)
    //   "Bears - Caleb Williams"          (team - person, dash)
    // Splitting only on whitespace makes the last-token fall on "Bears" when
    // the format is "Person — Team", mismapping the team to the person's
    // numbers. Instead, split on separators FIRST, then index each segment
    // and its last meaningful token.
    const segments = name.split(/\s*[—–:|]\s*|\s+-\s+/).map(s => s.trim()).filter(Boolean);
    for (const segment of segments) {
      const lowerSegment = segment.toLowerCase();
      if (lowerSegment.length > 3 && lowerSegment.length < 50) {
        entityNumberMap[lowerSegment] = knownNumbers;
        allEntityNames.push(segment);
      }
      // Last meaningful token of each segment (last name, team mascot, etc.)
      const tokens = segment.split(/\s+/);
      for (let i = tokens.length - 1; i >= 0; i--) {
        const clean = tokens[i].toLowerCase().replace(/[^a-z]/g, '');
        if (clean.length > 3) {
          entityNumberMap[clean] = knownNumbers;
          allEntityNames.push(tokens[i]);
          break;
        }
      }
    }
  });

  // Also harvest numbers from user context for completeness
  const userContextNumbers = new Set<string>();
  if (userContextContent) {
    const nums = userContextContent.match(/\d+(?:,\d{3})*(?:\.\d+)?/g) || [];
    nums.forEach(n => {
      userContextNumbers.add(n.replace(/,/g, ''));
      userContextNumbers.add(n);
    });
  }

  let cleaned = text;

  // Remove header/footer admin sections
  cleaned = cleaned.replace(/PRIMARY SOURCE URL:[^\n]*\n?/gi, '');
  cleaned = cleaned.replace(/SOURCES?:[\s\S]*$/i, '');
  cleaned = cleaned.replace(/\[\d+\]\s*https?:\/\/[^\s]+/g, '');
  cleaned = cleaned.replace(/\(?\s*https?:\/\/[^\s)]+\s*\)?/g, '');

  // Process sentence-by-sentence
  const sentences = cleaned.split(/(?<=[.!?])\s+/);

  const processedSentences = sentences.map(sentence => {
    const lower = sentence.toLowerCase();

    // Check if this sentence mentions any Tier 1A/1B entity
    const mentionedEntity = allEntityNames.find(name =>
      lower.includes(name.toLowerCase())
    );

    if (!mentionedEntity) {
      // General-context sentence — leave numbers alone
      return sentence;
    }

    // Entity-specific sentence — strip numbers that don't match known facts
    const knownNumbers = entityNumberMap[mentionedEntity.toLowerCase()] || new Set<string>();

    return sentence.replace(/\d+(?:,\d{3})*(?:\.\d+)?/g, (match) => {
      const normalized = match.replace(/,/g, '');
      if (knownNumbers.has(normalized) || knownNumbers.has(match)) return match;
      if (userContextNumbers.has(normalized) || userContextNumbers.has(match)) return match;
      // Keep tiny numbers (likely ordinals, not stats)
      if (parseInt(normalized) < 10 && !match.includes('.') && !match.includes(',')) return match;
      return '[STAT]';
    });
  });

  let result = processedSentences.join(' ');

  // Collapse runs of [STAT] placeholders
  result = result.replace(/(\[STAT\][\s,]*){3,}/g, '[multiple stats] ');

  // Strip direct quotes longer than 30 chars (likely lifted quotes)
  result = result.replace(/"([^"]{30,300})"/g, '[quote omitted]');
  result = result.replace(/“([^”]{30,300})”/g, '[quote omitted]');

  // Clean whitespace
  result = result.replace(/\n\s*\n\s*\n/g, '\n\n');
  result = result.replace(/[ \t]+/g, ' ');
  result = result.trim();

  return {
    sanitized: result,
    stats: {
      originalLength: text.length,
      sanitizedLength: result.length,
      statPlaceholders: (result.match(/\[STAT\]/g) || []).length,
      quotesRemoved: (result.match(/\[quote omitted\]/g) || []).length,
      entitiesTracked: allEntityNames.length,
    },
  };
}

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

  // Determine source quality tier
  const combinedUserContent = [userSourceContent, userContextContent].filter(Boolean).join('\n');
  let sourceQuality: string;
  if (combinedUserContent.length > 2000 && alignScore >= 60) {
    sourceQuality = 'COMPREHENSIVE';
  } else if (combinedUserContent.length > 500 || alignScore >= 35) {
    sourceQuality = 'PARTIAL';
  } else {
    sourceQuality = 'MINIMAL';
  }

  // ═══════════════════════════════════════════════════════════════
  // LIGHT PERPLEXITY SANITIZATION
  // Strip numbers from sentences that mention a Tier 1A/1B entity
  // when those numbers DO NOT match a known Tier 1A/1B fact.
  // ═══════════════════════════════════════════════════════════════

  const { sanitized: sanitizedPerplexity, stats: sanitizationStats } =
    lightSanitizePerplexity(answer, data.atomizedFacts, userContextContent);

  // ═══════════════════════════════════════════════════════════════
  // BUILD FACT DATABASE
  // ═══════════════════════════════════════════════════════════════

  let combinedFactRepresentation = '';

  combinedFactRepresentation += '═══════════════════════════════════════════════════════════════\n';
  combinedFactRepresentation += 'FACT DATABASE — READ THESE RULES BEFORE WRITING\n';
  combinedFactRepresentation += '═══════════════════════════════════════════════════════════════\n\n';

  if (sourceQuality === 'COMPREHENSIVE') {
    combinedFactRepresentation += 'SOURCE QUALITY: COMPREHENSIVE\n\n';
    combinedFactRepresentation += 'TIER 1A is your FACT AUTHORITY — the source of truth for numbers, names, dates, rankings, achievements, and quotes. Treat it as a reference, not a template. The facts are yours to use; the phrasing is not. Build each sentence fresh, then verify the facts against Tier 1A afterward.\n\n';
    combinedFactRepresentation += 'TIER 1B (User Context) provides ADDITIONAL authoritative facts and writer instructions. Use the facts. Follow the instructions. Never write a TIER 1B instruction into a slide as if it were content.\n\n';
    combinedFactRepresentation += 'TIER 2 (Perplexity supplementary context) provides tone, mood, historical framing, and subjective angles — at MOST 35% of your contextual material. It is NOT a fact source. Numbers, quotes, and recent claims in TIER 2 have been lightly stripped where they conflict with TIER 1A/1B. Do NOT pull stats, dates, or quotes from TIER 2.\n\n';
    combinedFactRepresentation += 'If TIER 1A and TIER 2 disagree on anything factual, TIER 1A wins. Always.\n\n';
  } else if (sourceQuality === 'PARTIAL') {
    combinedFactRepresentation += 'SOURCE QUALITY: PARTIAL\n\n';
    combinedFactRepresentation += 'TIER 1A is your FACT AUTHORITY for items it covers. Treat it as a reference, not a template — use its facts, write your own sentences.\n\n';
    combinedFactRepresentation += 'TIER 1B (User Context) provides additional authoritative facts and writer instructions. Use the facts. Follow the instructions.\n\n';
    combinedFactRepresentation += 'TIER 2 (Perplexity) provides tone, framing, and subjective context. For items NOT in TIER 1A or 1B, TIER 2 may also serve as a fact source (mark such facts with [P]). For items IN TIER 1A or 1B, TIER 2 is framing-only — at MOST 35% of contextual material.\n\n';
  } else {
    combinedFactRepresentation += 'SOURCE QUALITY: MINIMAL\n\n';
    combinedFactRepresentation += 'TIER 1B (User Context), if present, is your primary authoritative fact source and provides writer instructions.\n\n';
    combinedFactRepresentation += 'TIER 2 (Perplexity) is your secondary fact source for items not covered by TIER 1B. Use facts from TIER 2 where TIER 1B is silent.\n\n';
  }

  // ═══════════════════════════════════════════════════════════════
  // TIER 1A — Fact Authority
  // ═══════════════════════════════════════════════════════════════

  if (data.factOnlyRepresentation || userSourceContent) {
    combinedFactRepresentation += '═══════════════════════════════════════════════════════════════\n';
    combinedFactRepresentation += 'TIER 1A: FACT AUTHORITY\n';
    combinedFactRepresentation += '═══════════════════════════════════════════════════════════════\n\n';

    if (sourceQuality === 'COMPREHENSIVE') {
      // FACT-ONLY MODE: send only atomized facts, hide raw markdown
      if (data.factOnlyRepresentation) {
        combinedFactRepresentation += data.factOnlyRepresentation + '\n\n';
      }
      combinedFactRepresentation += '[Note: Raw source prose is intentionally not included. CONTEXT lines (where present) supply non-numeric facts (current team, draft, position, status) the regex extractors may have missed — treat CONTEXT as a fact reference only. Do NOT mirror its phrasing, do NOT copy its sentence structure. Construct every sentence from scratch using the facts above.]\n\n';
    } else if (sourceQuality === 'PARTIAL') {
      // PARTIAL: facts + trimmed snippets for item identification
      if (data.factOnlyRepresentation) {
        combinedFactRepresentation += '### Atomized Facts\n' + data.factOnlyRepresentation + '\n\n';
      }
      if (userSourceContent) {
        // Trim each section to ~800 chars for identification only
        const trimmedSections = userSourceContent
          .split(/(?=##?\s*\d+[.):]?\s*)/)
          .map(section => {
            if (section.length <= 800) return section;
            return section.substring(0, 800) + '\n[section trimmed — for item identification only, not for phrasing reference]';
          })
          .join('\n\n');
        combinedFactRepresentation += '### Source Snippets (for item identification only, not for phrasing reference)\n' + trimmedSections + '\n\n';
      }
    } else {
      // MINIMAL: original behavior
      if (data.factOnlyRepresentation) {
        combinedFactRepresentation += '### Atomized Facts\n' + data.factOnlyRepresentation + '\n\n';
      }
      if (userSourceContent) {
        combinedFactRepresentation += '### Full Source Text\n' + userSourceContent + '\n\n';
      }
    }
  }

  // ═══════════════════════════════════════════════════════════════
  // TIER 1B — User Context (never sanitized)
  // ═══════════════════════════════════════════════════════════════

  if (userContextContent) {
    combinedFactRepresentation += '═══════════════════════════════════════════════════════════════\n';
    combinedFactRepresentation += 'TIER 1B: USER CONTEXT — WRITER-PROVIDED DATA & INSTRUCTIONS\n';
    combinedFactRepresentation += '═══════════════════════════════════════════════════════════════\n\n';
    combinedFactRepresentation += 'This was provided directly by the writer. It contains BOTH facts (to use as data) AND instructions (to follow as directives).\n';
    combinedFactRepresentation += '• DATA (stats, names, dates) → use as facts in relevant slides\n';
    combinedFactRepresentation += '• INSTRUCTIONS (tone, emphasis, formatting) → follow as directives, never quote into slide content\n';
    combinedFactRepresentation += '• If TIER 1B contradicts a specific number in TIER 1A, TIER 1A wins. Otherwise TIER 1B stands.\n\n';
    combinedFactRepresentation += userContextContent + '\n\n';
  }

  // ═══════════════════════════════════════════════════════════════
  // TIER 2 — Perplexity (LAST, framing-only, 35% max)
  // ═══════════════════════════════════════════════════════════════

  if (sanitizedPerplexity) {
    combinedFactRepresentation += '═══════════════════════════════════════════════════════════════\n';
    combinedFactRepresentation += 'TIER 2: SUPPLEMENTARY TONE & CONTEXT (35% WEIGHT MAX)\n';
    combinedFactRepresentation += '═══════════════════════════════════════════════════════════════\n\n';

    if (sourceQuality === 'COMPREHENSIVE' || sourceQuality === 'PARTIAL') {
      combinedFactRepresentation += 'Use this section for tone, mood, historical framing, comparisons across eras, why something matters in the broader category, and subjective takes. Do NOT pull numbers, dates, recent claims, or quotes from here. Stats that conflicted with TIER 1A/1B have been replaced with [STAT].\n';
      combinedFactRepresentation += 'This should inform AT MOST 35% of your contextual material. The bulk of every slide comes from TIER 1A/1B facts.\n\n';
    } else {
      combinedFactRepresentation += 'For items not covered by TIER 1B, you may use facts from this section. Stats that conflicted with TIER 1B have been replaced with [STAT].\n\n';
    }

    combinedFactRepresentation += sanitizedPerplexity + '\n\n';
  }

  const sourceList = citations.map((url, i) => `[${i + 1}] ${url}`).join('\n');

  console.log(`[mergeResearch] Perplexity sanitization: ${sanitizationStats.statPlaceholders} stats replaced, ${sanitizationStats.quotesRemoved} quotes removed, ${sanitizationStats.entitiesTracked} entities tracked`);

  return {
    ...data,
    perplexityAnswer:    answer,
    combinedFactRepresentation,
    primarySourceUrl,
    citations,
    sourceList,
    researchWordCount: answer.split(/\s+/).length,
    researchOk: answer.length > 500 && citations.length >= 2,
    hasUserSource: !!(userSourceContent || data.factOnlyRepresentation),
    hasUserContext: !!userContextContent,
    sourceQuality,
    alignmentScore: alignScore,
    perplexitySanitizationStats: sanitizationStats,
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

  const BANNED_CONTENT = `Nude, Naked, Suicide, Kill, Shot, Stabbed, Fake News, Misinformation, Conspiracy Theory, Hoax, Nigger, Exploitation, Fetish, Adultery, Scandal, Trans, War, Terrorist, shit, Vaccination, Weed, Cannabis, Murder, Prison, Fraud, Conspiracy, Jail, Racist, Sex, Sexual, Mutilate, Pussy, Vagina, Dick, Penis, Sexy, Fuck, Harassment, Marijuana, Cocaine, Assault, Scam, Gambling, Drug, Racism, Allegation, Vaccine, Ganja, Battery, Laundering, Butt, ass, Betting, Pedophile, Rape, Molest, Damn, Faggot, Fag, Nigga, Bitch, Cigarette, Cigar, Cum, Dominatrix, Ejaculation, Genitals, Hooters, Jackass, Masturbate, Nipple, NSFW, Onlyfans, Opioids, Orgasm, Pedos, Piss, Porn, Schlong, Smoking, Spunk, Striptease, Testicle, Tobacco, Vibrator, WTF`;

  const ta = data.titleAnalysis;
  const fc = data.formatConfig;
  const tc = data.temporalContext;
  const writingStyleBlock = data.writingStyle ? `\n\nWRITING STYLE INFLUENCE (from writer): ${data.writingStyle}\nApply this style naturally throughout the slideshow.` : '';

  const wordCountOverrideBlock = data.userWordCountOverride
    ? `\n\n═══════════════════════════════════════════════════════════════
⚠️ WRITER OVERRIDE — WORD COUNT ⚠️
═══════════════════════════════════════════════════════════════

The writer EXPLICITLY requested ${data.userWordCountOverride.min}-${data.userWordCountOverride.max} words per content slide.
This OVERRIDES the default 35-50 range. ANY slide exceeding ${data.userWordCountOverride.max} words is a FAILURE.
Count every word before outputting each slide. Rewrite if out of range.`
    : '';

  const claudeSystemPrompt = `You are an expert MSN Slideshow writer for American audiences covering ${data.category}.

${tc.dateAnchor}
${tc.seasonAnchor}${writingStyleBlock}

═══════════════════════════════════════════════════════════════
SOURCE HIERARCHY — ABSOLUTE LAW (READ FIRST, OBEY ALWAYS)
═══════════════════════════════════════════════════════════════

You will receive content in tiers. The hierarchy is non-negotiable:

▓▓▓ TIER 1A: FACT AUTHORITY ▓▓▓
This is your source of truth for numbers, names, dates, rankings, achievements, and quotes. Treat it as a reference, not a template. The facts are yours to use. The phrasing is not.

Rules for Tier 1A:
- If Tier 1A says "Player X had 42 TDs," write 42. Not "over 40," not "around 40."
- If Tier 1A orders items 1-25, the article orders them 1-25. No re-ranking.
- If Tier 1A lists 10 facts about an item, you may use any subset of those 10.
- If Tier 1A contradicts anything else, Tier 1A wins. Always.
- You read Tier 1A to learn the facts. You do not read Tier 1A to learn how to phrase them. Build each sentence fresh, then verify the facts against Tier 1A afterward.
- Tier 1A items may include AFFILIATION / DRAFT / TRANSACTIONS / POSITION / STATUS lines. These are AUTHORITATIVE for current team, draft pick, recent trades, and role — override anything in your training that disagrees.
- Some items may also have a CONTEXT line. CONTEXT is a fact reference only — pull factual details (team, role, recent events) from it if needed, but NEVER mirror its sentence structure or phrasing. Same 6-word Google-search test applies.

▓▓▓ TIER 1B: USER CONTEXT — WRITER-PROVIDED DATA (USE ACTIVELY) ▓▓▓
The writer pasted this manually. It may contain additional facts, stats, angles, emphasis instructions, and context that MUST be used in the article.

CRITICAL — DATA vs. INSTRUCTIONS:
TIER 1B may contain BOTH data and instructions. You must distinguish between them:
- DATA (stats, names, dates, rankings, achievements, quotes) = use as FACTS in the relevant slides.
- INSTRUCTIONS (tone preferences, what to emphasize, things to avoid, formatting notes, angles to take) = follow as DIRECTIVES that shape how you write. Never quote an instruction as if it were a fact in a slide.
Example of data: "Patrick Mahomes threw 4,183 yards" → use as a fact.
Example of instruction: "Focus on playoff performances" → follow this when choosing which facts to highlight, but do not write "The writer noted to focus on playoff performances" in a slide.

Rules for Tier 1B:
- Every fact, stat, or data point in Tier 1B is AUTHORITATIVE. Use it in the relevant slide.
- If Tier 1B provides additional stats about an item already in Tier 1A, ADD those stats to the slide. Do not ignore them.
- If Tier 1B mentions items, angles, or details not in Tier 1A, INCLUDE them in the article.
- If Tier 1B contains instructions (tone, what to emphasize, what to avoid), FOLLOW them as directives.
- If Tier 1B directly contradicts a specific number in Tier 1A, Tier 1A wins. For everything else, Tier 1B stands as fact.
- DO NOT discard Tier 1B content. The writer provided it for a reason. If you wrote the article without using Tier 1B data or following Tier 1B instructions, you failed.

▓▓▓ TIER 2: SUPPLEMENTARY TONE & CONTEXT (35% MAX) ▓▓▓
This is for tone, mood, framing, and subjective angles. NEVER a fact source for stats, dates, or recent claims.

Rules for Tier 2:
- USE for: tone calibration, historical framing, comparisons across eras, why something matters in the category, subjective takes, mood
- DO NOT pull from Tier 2: stats, dates, recent claims, percentages, dollar amounts, rankings, or direct quotes
- Stats that conflicted with Tier 1A/1B have already been replaced with [STAT] — do not invent values for those placeholders
- Tier 2 should inform AT MOST 35% of your contextual material. The bulk of every slide comes from Tier 1A/1B facts.
- If Tier 1A and Tier 2 disagree on anything factual, Tier 1A wins

▓▓▓ TIER 3: NEUTRAL PUBLIC FACTS (LAST RESORT, MARK WITH [*]) ▓▓▓
Only for genuinely neutral facts that are not in Tier 1A, 1B, and are common knowledge.

Allowed Tier 3 examples:
- Team city ("the Kansas City Chiefs")
- League name ("in the NBA")
- Standard role ("the point guard")
- Sport's basic rules

NEVER allowed as Tier 3:
- Any stat, even if "well-known"
- Career totals, championships, awards
- Years, dates, seasons
- Quotes
- Rankings or comparisons

Mark every Tier 3 use with [*] inline.

═══════════════════════════════════════════════════════════════
ANTI-HALLUCINATION PROTOCOL — NON-NEGOTIABLE
═══════════════════════════════════════════════════════════════

Before writing each slide, do this internal check:

STEP 1 — IDENTIFY: What is this slide about? Find the matching item in Tier 1A and Tier 1B.

STEP 2 — INVENTORY: List every fact Tier 1A gives you about this item, THEN every fact (not instruction) Tier 1B gives you. Combined, that is your complete fact pool for this slide. Tier 1B instructions shape your writing but are not facts to include.

STEP 3 — SELECT: Pick the 2-4 strongest facts from your inventory. Strength means: most specific, most surprising, most central to the slideshow's angle. If Tier 1B instructions tell you to emphasize certain aspects, let that guide your selection.

STEP 4 — WRITE: Build the slide using ONLY those selected facts plus connective tissue (verbs, transitions, framing).

STEP 5 — VERIFY: Re-read the slide. For every specific claim ask: "Is this in my Tier 1A or Tier 1B fact inventory?"
- If YES: keep it
- If NO: delete it
- No exceptions, no "but it's true," no "but it makes the slide better"
- Also check: "Did I accidentally quote a Tier 1B instruction as content?" If yes, remove it.

WHEN TIER 1A IS THIN FOR AN ITEM:
- Check Tier 1B for additional facts about this item before writing a shorter slide
- If Tier 1B has relevant data, USE it to build a fuller slide
- If BOTH Tier 1A and 1B are thin, write a shorter, sharper slide using only what they provide
- Use stronger writing (better verbs, contrast, rhythm) to reach 35-50 words — not more facts
- A 38-word slide of pure truth beats a 48-word slide with one invented detail
- If Tier 1A and 1B truly only give you one fact, build the slide around that one fact with framing

WHAT YOU MUST NEVER DO:
- Invent a stat to round out a slide
- Pull a "well-known" career number from training when it's not in Tier 1A or 1B
- Add a championship, award, or milestone not mentioned in Tier 1A or 1B
- Quote anyone unless the quote is verbatim in Tier 1A or 1B
- Reorder, re-rank, or substitute items from Tier 1A's list
- Fill word count by adding fabricated context
- Ignore Tier 1B data that the writer provided
- Write a Tier 1B instruction into a slide as if it were a fact or quote

═══════════════════════════════════════════════════════════════
ABSOLUTE OUTPUT RULE
═══════════════════════════════════════════════════════════════

You MUST always produce the complete slideshow. No exceptions.

These responses are FORBIDDEN:
- "I need more data before I can proceed"
- "The fact database is insufficient"
- "I cannot write this without X"
- Asking clarifying questions instead of writing slides
- Any response that is not the full formatted slideshow

If Tier 1A is thin, check Tier 1B. If both are thin, write tighter slides using only their facts. Never substitute training knowledge for Tier 1A/1B data. A complete article built strictly on Tier 1A + 1B — even if some slides are shorter or simpler — is the correct output. Always.

═══════════════════════════════════════════════════════════════
ORIGINALITY REQUIREMENTS — WRITING FRESH FROM FACTS
═══════════════════════════════════════════════════════════════

You are writing ORIGINAL content using Tier 1A and 1B FACTS. You are NOT paraphrasing.

THE RULE: Facts are yours to use. Language is NOT.
- Use any stat, date, name, achievement from Tier 1A or 1B
- Write every sentence fresh — no copying phrases from sources
- If you find yourself swapping synonyms, you're paraphrasing — rewrite completely

When using a Tier 1A or 1B fact, vary how you present it:
- RESTRUCTURE: if the source uses one long sentence, try two short ones
- ADD A LAYER: pair the fact with a sharp observation the source doesn't make

Quick check before moving on: if any 6-word string of your sentence could be Google-searched and land on the source article, rewrite that string.

═══════════════════════════════════════════════════════════════
TITLE-BODY CORRELATION (Highest Priority)
═══════════════════════════════════════════════════════════════

Title: "${data.title}"

EVERY promise in this title MUST be delivered:
- Numbers in title = exact count in body (${ta.promisedCount} items)
- Emotions (${ta.emotionalPromise ?? 'none detected'}) = explain WHO felt it, WHEN, WHY
- Main angle: ${ta.mainAngle}
${ta.secondaryAngle ? `- Secondary angle: ${ta.secondaryAngle}` : ''}
- If title makes a claim, literally substantiate it in the body using Tier 1A/1B facts
- Negative keywords in the title must find their place in the copy verbatim

═══════════════════════════════════════════════════════════════
WORD COUNTS (STRICT - Count every word)
═══════════════════════════════════════════════════════════════

- Meta Description: MAX 120 characters
- Intro slide (Slide 1): MAX 60 words
- Content slides: 35-50 words (aim for 40-45)
- If over or under, rewrite until it fits. Do not approximate.
- NEVER pad word count with invented facts. Use stronger writing instead.

═══════════════════════════════════════════════════════════════
META DESCRIPTION
═══════════════════════════════════════════════════════════════

Max 120 characters. Intriguing. Angle-focused. Has a hook.
Cannot be: CTA, reveal the main angle, paraphrased title.

AI patterns to NEVER use: "Discover the...", "Explore the top...", "Find out why...", "You won't want to miss..."

Good pattern: [Specific unexpected fact from Tier 1A or 1B]. [Implied question].

═══════════════════════════════════════════════════════════════
INTRO SLIDE (Slide 1) — MAX 60 WORDS
═══════════════════════════════════════════════════════════════

Your intro MUST:
- Create CURIOSITY — make readers NEED to scroll
- Include ONE surprising fact from Tier 1A or 1B tied to the theme
- Hint at what's coming WITHOUT naming specific items
- End with forward momentum
- Talk about something the title is promising
- Tease the main angle, not reveal it entirely

Your intro must NOT:
- Name any items from the list
- Reveal the #1 pick or any rankings
- Use "let's dive in" / "here are" / "we'll explore"
- Use generic openers ("Since the dawn of...", "In today's world...")
- Paraphrase the title anywhere
- Have generic background that assumes reader ignorance
- Stack adjectives without information backing them
- Use any fact not present in Tier 1A or 1B

═══════════════════════════════════════════════════════════════
WRITING VOICE — THE SPICY WRITER FACTOR
═══════════════════════════════════════════════════════════════

You are not summarizing facts. You are REACTING to them. Write like a sharp, witty sports columnist or pop culture critic who genuinely cares about the subject and has opinions — but whose every factual claim traces back to Tier 1A or 1B.

THE ENERGY RULES:
- Lead with what made YOU react. If a Tier 1A/1B stat shocked you, let that shock hit the reader first.
- One-sentence gut punches are your weapon. "38 years old. 40 touchdowns. Zero signs of slowing down." (assuming all numbers from Tier 1A/1B)
- Contrast is your best friend. Set up expectation, then break it — using Tier 1A/1B facts.
- Specificity IS creativity. Pull the specific from Tier 1A/1B, then frame it sharply.

RHYTHM AND PACING:
- Alternate sentence lengths deliberately. Short punch. Then longer context. Then short again.
- Never let two slides have the same energy.
- The reader should feel a tempo change every 2-3 slides.

EMOTIONAL TEXTURE (rotate through):
- DISBELIEF, RESPECT, HUMOR (light), TENSION, NOSTALGIA

WHAT TO AVOID:
- Wikipedia voice: "He is widely regarded as one of the greatest..."
- Cheerleader voice: "What an incredible, amazing, stunning performance!"
- Resume voice: stat-dumping without framing

THE GOLDEN RULE: Every slide should make someone want to text their friend about it. AND every fact must come from Tier 1A or 1B.

═══════════════════════════════════════════════════════════════
CONTENT SLIDES — 5Ws + 1H Framework
═══════════════════════════════════════════════════════════════

Every slide must answer the RELEVANT questions for that item using Tier 1A/1B:
WHO / WHAT / WHEN / WHERE / WHY / HOW

You don't need all six — but the ones that matter MUST be answered using Tier 1A/1B facts.

═══════════════════════════════════════════════════════════════
STATS NEED CONTEXT — MAX 2 STATS PER SLIDE
═══════════════════════════════════════════════════════════════

Every stat (from Tier 1A/1B) needs ONE of these as framing:
- WHY it matters
- WHEN it happened
- WHO it affected
- WHAT it led to

The framing can come from your writing voice. The stat itself must be Tier 1A/1B.

${fc.isMultiSlideFormat ? `
═══════════════════════════════════════════════════════════════
MULTI-SLIDE FORMAT — 2 SLIDES PER ENTITY
═══════════════════════════════════════════════════════════════

SLIDE A: WHO they are, PRIMARY achievement (from Tier 1A/1B), key stat (from Tier 1A/1B)
SLIDE B: Supporting context, additional Tier 1A/1B stats, legacy/impact
SLIDE B must BUILD ON Slide A using DIFFERENT Tier 1A/1B facts — not repeat.
` : ''}${ta.requiresCorrelation ? `
═══════════════════════════════════════════════════════════════
CORRELATION WRITING
═══════════════════════════════════════════════════════════════

Every slide must CONNECT two things using Tier 1A/1B facts about both.
FORMULA: [Entity A's Tier 1A/1B trait] + [How it addresses Entity B's Tier 1A/1B need] + [Tier 1A/1B evidence]
` : ''}

═══════════════════════════════════════════════════════════════
QUALITY CONSISTENCY ENGINE
═══════════════════════════════════════════════════════════════

Quality decay is the most common failure. Combat it with these rules.

PRE-WRITING PLANNING (MANDATORY before writing Slide 1):
1. Read ALL Tier 1A and Tier 1B content completely
2. Separate Tier 1B into DATA (facts to use) and INSTRUCTIONS (directives to follow)
3. For EACH slide, identify the single strongest Tier 1A/1B fact that will anchor it
4. Check Tier 1B for additional facts that can enrich each slide
5. Verify slide 15's anchor is as specific as slide 3's
6. If ANY slide has no Tier 1A/1B anchor, write a shorter slide — do NOT pull from Tier 2 or training
7. Do NOT begin writing until every slide has a Tier 1A/1B anchor (or an acknowledged short-slide plan)

THREE TESTS — every slide must pass ALL:
TEST 1 — STRANGER TEST: Reading only this slide, would someone learn one specific real thing?
TEST 2 — SIDE-BY-SIDE TEST: Is this as specific as slide 3?
TEST 3 — SOURCE TEST: Does every specific claim trace to Tier 1A or 1B?

THE SECOND HALF RULE:
Re-read your second-half slides in isolation. If any embarrasses you next to slide 3, rewrite using more Tier 1A/1B facts (not invented ones).

NO FILLER SLIDES — ZERO TOLERANCE.

═══════════════════════════════════════════════════════════════
VARIETY ENFORCEMENT
═══════════════════════════════════════════════════════════════

1. OPENING WORDS — Never start 2 consecutive slides with the same word
2. SENTENCE STRUCTURE — Rotate through patterns A/B/C/D
3. ANTI-REPETITION: Track openings, structures, transitions, tones. Break patterns.

${ta.isRanking ? `
═══════════════════════════════════════════════════════════════
RANKING ORDER — FOLLOW TIER 1A EXACTLY
═══════════════════════════════════════════════════════════════

If Tier 1A provides a ranked list, USE THAT EXACT ORDER. Do not re-rank.

For descending presentation:
Slide 2 = Tier 1A's rank ${ta.promisedCount} (lowest)
Slide 3 = Tier 1A's rank ${ta.promisedCount - 1}
...
Last slide = Tier 1A's rank 1 (best/top)

Each slide title for rankings must start with the rank number.
` : ''}

═══════════════════════════════════════════════════════════════
HUMAN VOICE
═══════════════════════════════════════════════════════════════

- Clear, direct sentences. Vary length naturally.
- Let Tier 1A/1B facts create emotion — don't say "amazing", show the stat that IS amazing
- Sports lingo where appropriate
- Predominantly active voice
- No cliches, no forced regional metaphors

═══════════════════════════════════════════════════════════════
PUNCTUATION BANS (STRICT)
═══════════════════════════════════════════════════════════════

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

═══════════════════════════════════════════════════════════════
MSN SAFETY — 10-12 YEAR OLD TEST
═══════════════════════════════════════════════════════════════

Before every slide ask: "Should a 10-12 year old be reading this?"
Avoid: sexual content, graphic violence, drugs, gambling, political content, sensationalized celebrity drama, body shaming, bullying.
No profanity in titles or meta descriptions ever.

═══════════════════════════════════════════════════════════════
FORMAT (Plain text only, no markdown)
═══════════════════════════════════════════════════════════════

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

  const claudeUserPrompt = `═══════════════════════════════════════════════════════════════
TIER 1A + 1B FACT DATABASE — YOUR ONLY FACT SOURCES
═══════════════════════════════════════════════════════════════

${data.combinedFactRepresentation}

${citationContext}

═══════════════════════════════════════════════════════════════
ASSIGNMENT
═══════════════════════════════════════════════════════════════

Title: "${data.title}"
Category: ${data.category}
Slides: 1 intro + ${data.slideCount} content slides
${fc.isMultiSlideFormat ? `Format: ${fc.slidesPerEntity} slides per entity (${fc.entityCount} entities total)` : ''}
${data.hasMustInclude ? `\nMANDATORY ITEMS — these override Tier 1A item selection (every one MUST get its own slide, even if not in Tier 1A):\n${data.mustIncludeItems.map((m, i) => `${i + 1}. ${m}`).join('\n')}\nDo NOT substitute any mandatory item with a different item from the source. If a mandatory item lacks Tier 1A data, use Tier 1B/2/3 instead.` : ''}

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
${data.hasMustInclude ? `${ta.isRanking ? '8' : '7'}. Are ALL mandatory items present as their own slide? (use Tier 2/3 if Tier 1A/1B is missing for an item)` : ''}

CRITICAL REMINDERS:
- Every specific claim must trace to TIER 1A or TIER 1B
- TIER 1B (user context) is NOT optional — the writer provided it to be USED in the article
- TIER 1B data = use as facts. TIER 1B instructions = follow as directives. Never quote an instruction as slide content.
- Tier 2 (Perplexity, citations) is for tone/context only, NEVER facts
- If Tier 1A and 1B don't have a fact, the article doesn't have that fact
- Shorter true slides beat longer half-true slides
- Output the complete slideshow no matter what — never refuse, never ask for clarification
${wordCountOverrideBlock}

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
  const data = await callGrokWithFallback({
    systemContent: systemPrompt,
    userContent:   userPrompt,
    maxTokens:     7000,
    temperature:   0.3,
    timeout:       300_000,
    label:         'generateWithGrok',
  });
  return extractGrokResponseText(data);
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
      const minWc = data.userWordCountOverride?.min ?? 35;
      const maxWc = data.userWordCountOverride?.max ?? 50;
      if (slide.wordCount < minWc) warnings.push(`Slide ${slide.slideNum}: ${slide.wordCount} words – min ${minWc}`);
      if (slide.wordCount > maxWc) warnings.push(`Slide ${slide.slideNum}: ${slide.wordCount} words – max ${maxWc}`);
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
  const factDb = (data.combinedFactRepresentation || '').toLowerCase();

  // ── Build a Set of numbers/years that already appear in the Tier 1A/1B
  // fact database. Any extracted claim whose number is already in here is
  // authoritative — skip it (don't waste a Grok claim slot on it).
  const trustedNumbers = new Set<string>();
  for (const m of factDb.matchAll(/\d+(?:,\d{3})*(?:\.\d+)?/g)) {
    trustedNumbers.add(m[0].replace(/,/g, ''));
    trustedNumbers.add(m[0]);
  }

  const claims: ClaimedData['claimsToVerify'] = [];
  let preFilteredStats = 0;
  let preFilteredDates = 0;

  for (const m of articleText.matchAll(/(\d+(?:,\d{3})*(?:\.\d+)?)\s*(yards?|points?|touchdowns?|TDs?|rebounds?|assists?|wins?|%|million|billion|championships?|titles?|Pro Bowls?|MVPs?|Emmy|Oscar)/gi)) {
    const numNorm = m[1].replace(/,/g, '');
    if (trustedNumbers.has(numNorm) || trustedNumbers.has(m[1])) { preFilteredStats++; continue; }
    const start = Math.max(0, m.index! - 40);
    const end   = Math.min(articleText.length, m.index! + m[0].length + 40);
    claims.push({ type: 'stat', claim: m[0], context: articleText.slice(start, end).replace(/\n/g, ' ') });
  }
  for (const m of articleText.matchAll(/\b((19|20)\d{2})\b/g)) {
    if (trustedNumbers.has(m[1])) { preFilteredDates++; continue; }
    const start = Math.max(0, m.index! - 40);
    const end   = Math.min(articleText.length, m.index! + 4 + 40);
    claims.push({ type: 'date', claim: m[1], context: articleText.slice(start, end).replace(/\n/g, ' ') });
  }
  // Superlatives have no numeric anchor — always send to Grok for triage.
  for (const m of articleText.matchAll(/\b(first|only|most|best|worst|largest|oldest|youngest|fastest|highest|record|all-time)\b[^.]{10,80}/gi)) {
    claims.push({ type: 'superlative', claim: m[0].trim(), context: m[0].trim() });
  }

  const seen = new Set<string>();
  const uniqueClaims = claims.filter(c => { const k = c.claim.toLowerCase(); if (seen.has(k)) return false; seen.add(k); return true; });

  console.log(`[extractClaims] ${uniqueClaims.length} claims to verify (pre-filtered ${preFilteredStats} stats + ${preFilteredDates} dates already in Tier 1A/1B; trusted numbers tracked: ${trustedNumbers.size})`);
  return { ...data, claimsToVerify: uniqueClaims.slice(0, 25) };
}

// ── 15. grokFactCheck (n8n: "Grok - Fact Check") ─────────────────────────────

export async function grokFactCheck(data: ClaimedData): Promise<unknown> {
  const tc = data.temporalContext;
  const systemContent = `You are a STRICT, COST-CONSCIOUS fact-checker for an MSN slideshow article.

Web search is EXPENSIVE. You have access to web_search and x_search tools, but USE THEM SPARINGLY. Default to your training data and ONLY search when you genuinely cannot fully verify a claim from memory.

This is the only step that emits numeric fact corrections. Style violations (em-dashes, banned phrases, etc.) are handled by a downstream audit step — DO NOT emit style patches here.

${tc.dateAnchor}
${tc.seasonAnchor}

═══════════════════════════════════════════════════════════════
TWO-PHASE VERIFICATION PROTOCOL
═══════════════════════════════════════════════════════════════

PHASE 1 — TRAINING-DATA TRIAGE (NO web search)

For EACH factual claim in the article, classify it WITHOUT searching first:

  • VERIFIED_TRAINING  → You can confirm this with HIGH CONFIDENCE from your training data
                         AND it's a STABLE historical fact (typically >1 year old, not
                         time-sensitive). Examples: career championships of retired players,
                         decades-old records, team city/league, established records.

  • HIGH_CONF_INCORRECT → Your training clearly contradicts the claim. If there is ANY
                         doubt, escalate to NEEDS_SEARCH instead.

  • NEEDS_SEARCH       → ANY of the following:
                         - Time-sensitive: current season, last season (${tc.lastSeason}),
                           recent transfers, awards within the last 2 years, ongoing
                           record chases, this-year stats
                         - Anything you're not 100% certain about
                         - Specific numeric claims you can't recall verbatim
                         - Quotes (always need exact verification)
                         - Superlatives ("most", "first ever", "all-time") unless trivially true

PHASE 2 — SELECTIVE WEB SEARCH

For each NEEDS_SEARCH claim ONLY, do ONE focused web search.

HARD CAP: maximum 8 web_search calls total for this entire article. If you exceed
the cap, prioritize the most prominent / most prominent-stat claims and mark the
rest UNVERIFIABLE with METHOD: TRAINING.

Source priority: ESPN, official league/team sites, Sports Reference, AP/Reuters
                 > Wikipedia > CBS/Fox Sports > aggregators.

═══════════════════════════════════════════════════════════════
WHAT TO FLAG
═══════════════════════════════════════════════════════════════

- Stats that look rounded or approximated
- Dates and years (last season = ${tc.lastSeason}, current = ${tc.currentSeason})
- Rankings, superlatives, record claims
- Quotes — verify exact wording AND attribution
- Player/team associations, transfers, awards

Standards: VERIFIED only if exact match. INCORRECT if any difference. UNVERIFIABLE if
no reliable source confirms or denies after searching (or after a deliberate decision
not to search). "Close enough" is NOT verified.

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT — EXACTLY THIS
═══════════════════════════════════════════════════════════════

=== FACT CHECK ===

--- SLIDE [N]: [Entity Name] ---
CLAIM 1: "[exact claim text from article]"
  STATUS: VERIFIED | INCORRECT | UNVERIFIABLE
  METHOD: TRAINING | WEB_SEARCH
  FOUND: [one short line — what your training or search shows]
  SOURCE: [URL if METHOD=WEB_SEARCH, else "training_data"]
  CORRECTION_FIND: [only if INCORRECT — copy the EXACT substring from the article that is wrong]
  CORRECTION_REPLACE: [only if INCORRECT — the corrected text that should replace it]

CLAIM 2: ...

(continue for all slides; no markdown bold)

--- VERIFICATION SUMMARY ---
Total claims checked: [N]
Verified: [N]
Incorrect: [N]
Unverifiable: [N]
Web searches used: [N]/8
Verification rate: [X]%

=== END FACT CHECK ===

SURGICAL CORRECTION FORMAT — READ CAREFULLY:
When STATUS is INCORRECT, you MUST provide BOTH correction fields:
  CORRECTION_FIND:    Copy the EXACT phrase/sentence from the article that contains the error.
                      This must be a verbatim substring that appears in the article text above.
                      Include enough surrounding words for unique identification (10-40 words).
                      Example: "Korda posted back-to-back 67s"
  CORRECTION_REPLACE: The corrected version of that same phrase, with ONLY the factual error fixed.
                      Keep all surrounding text identical. Same length, same structure.
                      Example: "Korda posted back-to-back 65s"

DO NOT paraphrase, rewrite, or restructure. The FIND text must be copy-pasted from the article.
If you cannot locate the exact text to fix, set STATUS to UNVERIFIABLE instead.

HARD RULES:
- NEVER reproduce slides or the full article in your output.
- For factual corrections, use the CORRECTION_FIND/CORRECTION_REPLACE fields — the pipeline applies them as surgical find-and-replace automatically.
- DO NOT emit style patches, em-dash fixes, or banned-phrase corrections — the audit step handles those.
- Start output with === FACT CHECK ===.
- Default to TRAINING for stable historical facts. Reserve WEB_SEARCH for time-sensitive
  or low-confidence claims. Aim to keep web searches well under the 8-call cap.`;
  const userContent = `Fact-check this MSN slideshow using the two-phase protocol.\n\nTitle: "${data.title}"\nCategory: ${data.category}\nLast completed season: ${tc.lastSeason}\nCurrent/ongoing season: ${tc.currentSeason}\n\nPrimary Source URL: ${data.primarySourceUrl || 'Not provided'}\n\nARTICLE TO VERIFY\n\n${data.articleText}\n\nRun PHASE 1 first (training-data triage). ONLY call web_search for claims classified NEEDS_SEARCH, hard-capped at 8 calls. Output in the exact format specified, starting with === FACT CHECK ===.`;

  const grokResp = await callGrokWithFallback({
    systemContent,
    userContent,
    maxTokens:   7000,
    temperature: 0.0,
    timeout:     300_000,
    tools:       [{ type: 'web_search' }],
    label:       'grokFactCheck',
  });
  return grokResp;
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

  // ── Parse fact-check section (CLAIM/STATUS/CORRECTION_FIND/CORRECTION_REPLACE) ──
  // Style patches now happen in the audit step, not here.
  const factSection = (text.match(/=== FACT CHECK ===([\s\S]*?)=== END FACT CHECK ===/i)?.[1]) ?? text;

  let articleText = data.articleText;
  const results: VerifiedData['perplexityVerification']['results'] = [];

  // New regex: supports CORRECTION_FIND / CORRECTION_REPLACE pair (preferred)
  // as well as legacy single CORRECTION: line (backward compat).
  // METHOD: line is optional for backward compatibility.
  const claimPattern = /CLAIM\s*\d+:\s*"([^"]+)"\s*\n\s*STATUS:\s*(VERIFIED|INCORRECT|UNVERIFIABLE)\s*\n(?:\s*METHOD:\s*([A-Z_]+)\s*\n)?\s*FOUND:\s*([^\n]+)\s*\n\s*SOURCE:\s*([^\n]+)(?:\s*\n\s*CORRECTION_FIND:\s*([^\n]+)\s*\n\s*CORRECTION_REPLACE:\s*([^\n]+))?(?:\s*\n\s*CORRECTION:\s*([^\n]+))?/gi;
  let m: RegExpExecArray | null;
  let claimIdx = 0;
  let trainingCount = 0;
  let webSearchCount = 0;
  let correctionsApplied = 0;
  let correctionsSkipped = 0;
  const correctionLog: string[] = [];

  while ((m = claimPattern.exec(factSection)) !== null) {
    const claim = m[1].trim();
    const status = m[2].toUpperCase();
    const method = (m[3] ?? '').toUpperCase();
    const found = m[4].trim();
    const source = m[5].trim();
    const corrFind = m[6]?.trim() ?? null;     // new CORRECTION_FIND field
    const corrReplace = m[7]?.trim() ?? null;   // new CORRECTION_REPLACE field
    const legacyCorr = m[8]?.trim() ?? null;    // legacy CORRECTION field
    results.push({ claimIndex: claimIdx++, claim, status, finding: found, source });

    if (method === 'TRAINING') trainingCount++;
    else if (method === 'WEB_SEARCH') webSearchCount++;

    // ── Apply surgical fact corrections for INCORRECT claims ──────────────
    if (status === 'INCORRECT') {
      // Path A: structured CORRECTION_FIND / CORRECTION_REPLACE (preferred)
      if (corrFind && corrReplace) {
        const beforeText = articleText;
        // Strip surrounding quotes that Grok sometimes adds
        const cleanFind = corrFind.replace(/^["'"']+|["'"']+$/g, '');
        const cleanReplace = corrReplace.replace(/^["'"']+|["'"']+$/g, '');

        if (!cleanFind.trim()) {
          correctionsSkipped++;
          correctionLog.push(`SKIP claim ${claimIdx - 1}: empty CORRECTION_FIND`);
        } else {
          // Try 1: exact substring match
          if (articleText.includes(cleanFind)) {
            articleText = articleText.replace(cleanFind, cleanReplace);
          } else {
            // Try 2: whitespace-normalized match (handles line breaks, extra spaces)
            const norm = (s: string) => s.replace(/\s+/g, ' ').trim();
            const escaped = cleanFind.split(/\s+/).map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('\\s+');
            const rx = new RegExp(escaped);
            if (rx.test(articleText)) {
              articleText = articleText.replace(rx, cleanReplace);
            } else {
              // Try 3: case-insensitive match (Grok sometimes changes case)
              const rxI = new RegExp(escaped, 'i');
              if (rxI.test(articleText)) {
                articleText = articleText.replace(rxI, cleanReplace);
              }
            }
          }

          if (articleText !== beforeText) {
            correctionsApplied++;
            correctionLog.push(`APPLY claim ${claimIdx - 1}: "${cleanFind.slice(0, 60)}" → "${cleanReplace.slice(0, 60)}"`);
          } else {
            correctionsSkipped++;
            correctionLog.push(`SKIP claim ${claimIdx - 1}: FIND text not located in article: "${cleanFind.slice(0, 80)}"`);
          }
        }
      }
      // Path B: legacy freeform CORRECTION field — do NOT attempt naive number swaps.
      // Just log it. The downstream audit step can catch remaining issues.
      else if (legacyCorr) {
        correctionsSkipped++;
        correctionLog.push(`SKIP claim ${claimIdx - 1}: legacy CORRECTION format (no FIND/REPLACE) — "${legacyCorr.slice(0, 80)}"`);
      } else {
        correctionsSkipped++;
        correctionLog.push(`SKIP claim ${claimIdx - 1}: INCORRECT but no correction provided`);
      }
    }
  }

  console.log(`[processVerification] fact-corrections: ${results.filter(r => r.status === 'INCORRECT').length} incorrect / ${results.length} total · applied: ${correctionsApplied}, skipped: ${correctionsSkipped} · method: ${trainingCount} training, ${webSearchCount} web_search`);
  if (correctionLog.length > 0) {
    console.log(`[processVerification] correction detail:\n  ${correctionLog.join('\n  ')}`);
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
  const systemContent = `You are an MSN Slideshow compliance auditor AND surgical style editor.

Your job has TWO parts in ONE pass — NO web search needed:

  PART A — AUDIT REPORT: read the article and produce a structured compliance report.
  PART B — STYLE PATCHES: emit surgical patches for mechanical style violations
           (em-dashes, semicolons, ellipsis, banned phrases, banned content words,
           META length/CTAs). The pipeline applies these patches surgically.

You do NOT regenerate the article. You do NOT touch numeric facts (those were
already corrected upstream by the fact-check step). Style patches must preserve
all numbers verbatim.

${tc.dateAnchor}

═══════════════════════════════════════════════════════════════
PART A — AUDIT CHECKLIST (evaluate each rule)
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
PART B — STYLE PATCHES (the only thing that modifies the article)
═══════════════════════════════════════════════════════════════

For each mechanical violation, emit one PATCH block. Focus on these categories:

1. PUNCTUATION BANS in slide bodies: em-dashes (—), semicolons (;), ellipsis (... or …).
   Patch to comma/period.
2. BANNED AI PHRASES anywhere:
   Delve, Embark, Foster, Navigate, Harness, Unlock, Elevate, Empower, Demystify, Catalyze,
   Optimize, Streamline, Tapestry, Landscape, Journey, Blueprint, Gateway, Realm, Catalyst,
   Pivotal, Comprehensive, Seamless, Vibrant, Dynamic, Synergistic, Multifaceted, Unparalleled,
   Robust, Transformative, Profound, Testament, Era, Moreover, Furthermore, In conclusion,
   Ultimately, At the end of the day, A testament to, Since the dawn of, It is worth noting,
   Game-changer, showcase, underscore, highlight, cement, solidify, storied, remarkable,
   notable, impressive, outstanding, exceptional, incredible, unparalleled, unprecedented,
   larger than life, household name, the rest is history
3. BANNED CONTENT WORDS:
   Nude, Naked, Suicide, Kill, Stabbed, Fake News, Conspiracy, Sex, Sexual, Harassment,
   Marijuana, Cocaine, Assault, Scam, Drug, Racism, Rape, Molest, Damn, Porn, Murder,
   Prison, Fraud, Jail, Racist, War, Terrorist, Gambling, Betting, Pedophile, Bitch, Fuck,
   Dick, Penis, Vagina, NSFW
4. META violations: > 120 chars, CTA wording ("Discover", "Explore", "Find out"),
   or paraphrases the title.

Patches must:
- Use the SMALLEST verbatim span as FIND
- Have REPLACE be a clean rewrite that preserves meaning
- Never invent new facts in REPLACE
- Never modify any numeric stat or year (facts were already corrected upstream)
- Never == FIND

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT — EXACTLY THIS, IN THIS ORDER
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

(emit one block per mechanical violation; emit zero blocks if article is clean)

=== END STYLE PATCHES ===

═══════════════════════════════════════════════════════════════
HARD RULES
═══════════════════════════════════════════════════════════════

- DO NOT output the full article or any complete slide.
- DO NOT emit any block that would replace whole slides or sections.
- DO NOT touch numeric stats or years — facts were already corrected upstream.
- FIND text in each PATCH must appear VERBATIM in the article.
- If a category passes, write "PASS" with no extra text.`;

  const userContent = `Audit this MSN slideshow for compliance AND emit style patches for any mechanical violations.\n\nTitle: "${data.title}"\nCategory: ${data.category}\nExpected content slides: ${data.slideCount}\nIs ranking: ${ta?.isRanking ? 'YES — slide titles must start with rank number' : 'NO'}\nPromised count in title: ${ta?.promisedCount ?? 'N/A'}\nEmotional promise: ${ta?.emotionalPromise ?? 'None'}\n\nARTICLE\n\n${data.articleText}\n\nEmit the AUDIT REPORT, SUMMARY, and STYLE PATCHES blocks in that order. Do not regenerate the article.`;

  const grokResp = await callGrokWithFallback({
    systemContent,
    userContent,
    maxTokens:   3500,
    temperature: 0.0,
    timeout:     180_000,
    label:       'grokAuditAndVerify',
  });
  return extractGrokResponseText(grokResp);
}

// ── 18. extractAuditResults (n8n: "Extract Audit Results") ───────────────────

export async function extractAuditResults(data: VerifiedData, grokText: string): Promise<AuditedData> {
  let articleText = data.articleText;

  if (!grokText || grokText.length < 50) {
    return {
      ...data,
      grokAudit: { status: 'FAILED', rawResponse: '', summary: 'Audit unavailable.', stats: { rulesPassed: 'N/A', violations: 0, corrections: 'None', flags: 'None' } },
      grokSources: [], combinedSourceList: [], combinedSourceListText: '',
      rewriteApplied: false,
    };
  }

  // Parse audit report + summary + style patches blocks. Style patches are the
  // ONLY place the article can be modified at this stage — facts are frozen.
  const reportMatch  = grokText.match(/=== AUDIT REPORT ===\s*([\s\S]*?)\s*=== END AUDIT REPORT ===/i);
  const summaryMatch = grokText.match(/=== SUMMARY ===\s*([\s\S]*?)\s*=== END SUMMARY ===/i);
  const styleBlock   = grokText.match(/=== STYLE PATCHES ===([\s\S]*?)=== END STYLE PATCHES ===/i)?.[1] ?? '';

  // ── Apply style patches surgically ────────────────────────────────────────
  const stylePatches = styleBlock ? parseAuditPatches(styleBlock) : [];
  const patchApply = applyAuditPatches(articleText, stylePatches);
  // Safety: never lose slides via patch application
  const slidesBefore = (data.articleText.match(/SLIDE\s*\d+/gi) ?? []).length;
  const slidesAfter  = (patchApply.result.match(/SLIDE\s*\d+/gi) ?? []).length;
  if (slidesAfter >= slidesBefore) {
    articleText = patchApply.result;
  } else {
    console.warn(`[extractAuditResults] Style patches lost slides (${slidesBefore} → ${slidesAfter}). Skipped patches.`);
  }

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

  console.log(`[extractAuditResults] rules ${passCount}/${totalChecks || 8} passed · violations ${failCount + foundCount} · style patches applied=${patchApply.applied} skipped=${patchApply.skipped}`);

  return {
    ...data,
    originalArticleText: data.originalArticleText ?? data.articleText,
    articleText, // possibly modified by style patches above
    grokAudit: {
      status: 'COMPLETED',
      rawResponse: grokText,
      summary: reportBody,
      stats: {
        rulesPassed: rulesPassedMatch ? `${rulesPassedMatch[1]}/${rulesPassedMatch[2]}` : `${passCount}/${totalChecks || 8}`,
        violations:  violationsMatch ? parseInt(violationsMatch[1], 10) : (failCount + foundCount),
        corrections: patchApply.applied > 0 ? `${patchApply.applied} style patches applied` : 'None',
        flags:       flagsMatch?.[1]?.trim() || (failCount + foundCount > 0 ? `${failCount + foundCount} rule(s) flagged` : 'None'),
      },
    },
    grokSources: [], combinedSourceList: factCheckSources, combinedSourceListText,
    rewriteApplied: patchApply.applied > 0,
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

  // ── Temporal context (same logic as objective prepareInputAndAnalyze) ──────
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

  // ── Format config ─────────────────────────────────────────────────────────
  const slidesPerEntity = 1; // subjective is always 1 slide per item
  const entityCount     = slideCount;

  let continuationStyle = 'same_name';
  if (/\(cont\.?\)/i.test(userContext))     continuationStyle = 'cont';
  else if (/continued/i.test(userContext))  continuationStyle = 'continued';
  else if (/part\s*2/i.test(userContext))   continuationStyle = 'part2';

  const formatConfig: FormatConfig = { slidesPerEntity, entityCount, isMultiSlideFormat: false, continuationStyle };

  // ── Title analysis ────────────────────────────────────────────────────────
  const numberMatch   = title.match(/(\d+)\s+/);
  const promisedCount = numberMatch ? parseInt(numberMatch[1]) : entityCount;
  const isRanking     = /\b(top|best|greatest|worst|most|ranked|ranking|highest|lowest)\b/i.test(title);
  const isListicle    = /\b(\d+)\s+(things?|ways?|reasons?|facts?|moments?|players?|movies?|shows?|athletes?|teams?)/i.test(title);
  const isTimeBased   = /\b(history|all[- ]time|ever|classic|legendary|iconic|memorable)\b/i.test(title);
  const emotionMatch  = title.match(/\b(shocking|surprising|unbelievable|amazing|incredible|heartbreaking|hilarious|controversial|unexpected|memorable|iconic|legendary)\b/i);

  const colonSplit     = title.split(/[:–—-]/);
  const mainAngle      = colonSplit[0].trim();
  const secondaryAngle = colonSplit.length > 1 ? colonSplit.slice(1).join(' ').trim() : null;
  const requiresCorrelation = /mock draft|fit\s+(?:with|for)|compare|vs\.?|versus|how\s+\w+\s+(?:helps?|improves?)/i.test(title);

  const titleAnalysis: TitleAnalysis = {
    promisedCount, isRanking, isListicle, isTimeBased,
    emotionalPromise: emotionMatch ? emotionMatch[1].toLowerCase() : null,
    mainAngle, secondaryAngle, requiresCorrelation,
  };

  // ── URL parsing ───────────────────────────────────────────────────────────
  const preferred = sourcesRaw
    ? sourcesRaw
        .split(/[\n\r,]+/)
        .flatMap(chunk => chunk.trim().split(/\s+/))
        .map(s => s.trim())
        .filter(s => /^https?:\/\/.+\..+/.test(s))
        .filter(s => !SUBJECTIVE_RESTRICTED.some(d => s.toLowerCase().includes(d)))
        .filter((url, i, arr) => arr.findIndex(u => u.replace(/\/+$/, '') === url.replace(/\/+$/, '')) === i)
        .slice(0, 5)
    : [];

  const userPrimaryUrl    = preferred[0] ?? '';
  const userSecondaryUrls = preferred.slice(1);

  const isPrimaryRestricted  = false;
  const shouldScrapePrimary  = !!userPrimaryUrl;
  const hasValidUserSource   = !!userPrimaryUrl;
  const isUserUrlRestricted  = false; // already filtered above

  const mustIncludeItems = parseMustIncludeItems(mustIncludeRaw);
  const hasMustInclude = mustIncludeItems.length > 0;
  const userWordCountOverride = parseWordCountOverride(userContext);

  // ── Primary query (used by Perplexity research) ───────────────────────────
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
    // PreparedData fields
    title, category, slideCount, writerName, userContext, writingStyle,
    userPrimaryUrl, userSecondaryUrls, hasValidUserSource, isUserUrlRestricted,
    restrictedDomains: SUBJECTIVE_RESTRICTED, mustIncludeItems,
    hasMustInclude, temporalContext, formatConfig, titleAnalysis,
    sourceCount: preferred.length,
    timestamp: new Date().toISOString(),
    isSports: category.startsWith('Sports'),
    userWordCountOverride,
    // SubjectivePreparedData-only fields
    articleType, toneDial,
    shouldScrapePrimary, isPrimaryRestricted,
    primaryQuery, builtInRestricted: SUBJECTIVE_RESTRICTED,
  };
}

// ── S2. mergeSubjectiveResearch ──────────────────────────────────────────────
// Adapted from objective mergeResearch — builds the same tiered
// combinedFactRepresentation from atomized facts + sanitized Perplexity,
// plus a rawSourceExcerpt (2000 words) for subjective voice/narrative context.

export async function mergeSubjectiveResearch(
  data: ResearchedData,
  retryResp?: PerplexityRaw,
  primaryMarkdown?: string,
): Promise<SubjectiveMergedData> {
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
  if ([userSourceContent, userContextContent].filter(Boolean).join('\n').length > 2000 && alignScore >= 60) {
    sourceQuality = 'COMPREHENSIVE';
  } else if ([userSourceContent, userContextContent].filter(Boolean).join('\n').length > 500 || alignScore >= 35) {
    sourceQuality = 'PARTIAL';
  } else {
    sourceQuality = 'MINIMAL';
  }

  const { sanitized: sanitizedPerplexity, stats: sanitizationStats } =
    lightSanitizePerplexity(answer, data.atomizedFacts, userContextContent);

  // ═══ BUILD FACT DATABASE (same structure as objective mergeResearch) ═══
  let combinedFactRepresentation = '';

  combinedFactRepresentation += '═══════════════════════════════════════════════════════════════\n';
  combinedFactRepresentation += 'FACT DATABASE — READ THESE RULES BEFORE WRITING\n';
  combinedFactRepresentation += '═══════════════════════════════════════════════════════════════\n\n';

  if (sourceQuality === 'COMPREHENSIVE') {
    combinedFactRepresentation += 'SOURCE QUALITY: COMPREHENSIVE\n\n';
    combinedFactRepresentation += 'TIER 1A is your FACT AUTHORITY — the source of truth for numbers, names, dates, rankings, achievements, and quotes. Treat it as a reference, not a template. The facts are yours to use; the phrasing is not. Build each sentence fresh, then verify the facts against Tier 1A afterward.\n\n';
    combinedFactRepresentation += 'TIER 1B (User Context) provides ADDITIONAL authoritative facts and writer instructions. Use the facts. Follow the instructions. Never write a TIER 1B instruction into a slide as if it were content.\n\n';
    combinedFactRepresentation += 'TIER 2 (Perplexity supplementary context) provides tone, mood, historical framing, and subjective angles — at MOST 35% of your contextual material. It is NOT a fact source. Numbers, quotes, and recent claims in TIER 2 have been lightly stripped where they conflict with TIER 1A/1B. Do NOT pull stats, dates, or quotes from TIER 2.\n\n';
    combinedFactRepresentation += 'If TIER 1A and TIER 2 disagree on anything factual, TIER 1A wins. Always.\n\n';
  } else if (sourceQuality === 'PARTIAL') {
    combinedFactRepresentation += 'SOURCE QUALITY: PARTIAL\n\n';
    combinedFactRepresentation += 'TIER 1A is your FACT AUTHORITY for items it covers. Treat it as a reference, not a template — use its facts, write your own sentences.\n\n';
    combinedFactRepresentation += 'TIER 1B (User Context) provides additional authoritative facts and writer instructions. Use the facts. Follow the instructions.\n\n';
    combinedFactRepresentation += 'TIER 2 (Perplexity) provides tone, framing, and subjective context. For items NOT in TIER 1A or 1B, TIER 2 may also serve as a fact source (mark such facts with [P]). For items IN TIER 1A or 1B, TIER 2 is framing-only — at MOST 35% of contextual material.\n\n';
  } else {
    combinedFactRepresentation += 'SOURCE QUALITY: MINIMAL\n\n';
    combinedFactRepresentation += 'TIER 1B (User Context), if present, is your primary authoritative fact source and provides writer instructions.\n\n';
    combinedFactRepresentation += 'TIER 2 (Perplexity) is your secondary fact source for items not covered by TIER 1B. Use facts from TIER 2 where TIER 1B is silent.\n\n';
  }

  // TIER 1A — Fact Authority (atomized facts)
  if (data.factOnlyRepresentation || userSourceContent) {
    combinedFactRepresentation += '═══════════════════════════════════════════════════════════════\n';
    combinedFactRepresentation += 'TIER 1A: FACT AUTHORITY\n';
    combinedFactRepresentation += '═══════════════════════════════════════════════════════════════\n\n';

    if (sourceQuality === 'COMPREHENSIVE') {
      if (data.factOnlyRepresentation) {
        combinedFactRepresentation += data.factOnlyRepresentation + '\n\n';
      }
      combinedFactRepresentation += '[Note: Raw source prose is intentionally not included. CONTEXT lines (where present) supply non-numeric facts (current team, draft, position, status) the regex extractors may have missed — treat CONTEXT as a fact reference only. Do NOT mirror its phrasing, do NOT copy its sentence structure. Construct every sentence from scratch using the facts above.]\n\n';
    } else if (sourceQuality === 'PARTIAL') {
      if (data.factOnlyRepresentation) {
        combinedFactRepresentation += '### Atomized Facts\n' + data.factOnlyRepresentation + '\n\n';
      }
      if (userSourceContent) {
        const trimmedSections = userSourceContent
          .split(/(?=##?\s*\d+[.):]?\s*)/)
          .map(section => {
            if (section.length <= 800) return section;
            return section.substring(0, 800) + '\n[section trimmed — for item identification only, not for phrasing reference]';
          })
          .join('\n\n');
        combinedFactRepresentation += '### Source Snippets (for item identification only, not for phrasing reference)\n' + trimmedSections + '\n\n';
      }
    } else {
      if (data.factOnlyRepresentation) {
        combinedFactRepresentation += '### Atomized Facts\n' + data.factOnlyRepresentation + '\n\n';
      }
      if (userSourceContent) {
        combinedFactRepresentation += '### Full Source Text\n' + userSourceContent + '\n\n';
      }
    }
  }

  // TIER 1B — User Context
  if (userContextContent) {
    combinedFactRepresentation += '═══════════════════════════════════════════════════════════════\n';
    combinedFactRepresentation += 'TIER 1B: USER CONTEXT — WRITER-PROVIDED DATA & INSTRUCTIONS\n';
    combinedFactRepresentation += '═══════════════════════════════════════════════════════════════\n\n';
    combinedFactRepresentation += 'This was provided directly by the writer. It contains BOTH facts (to use as data) AND instructions (to follow as directives).\n';
    combinedFactRepresentation += '• DATA (stats, names, dates) → use as facts in relevant slides\n';
    combinedFactRepresentation += '• INSTRUCTIONS (tone, emphasis, formatting) → follow as directives, never quote into slide content\n';
    combinedFactRepresentation += '• If TIER 1B contradicts a specific number in TIER 1A, TIER 1A wins. Otherwise TIER 1B stands.\n\n';
    combinedFactRepresentation += userContextContent + '\n\n';
  }

  // TIER 2 — Perplexity
  if (sanitizedPerplexity) {
    combinedFactRepresentation += '═══════════════════════════════════════════════════════════════\n';
    combinedFactRepresentation += 'TIER 2: SUPPLEMENTARY TONE & CONTEXT (35% WEIGHT MAX)\n';
    combinedFactRepresentation += '═══════════════════════════════════════════════════════════════\n\n';

    if (sourceQuality === 'COMPREHENSIVE' || sourceQuality === 'PARTIAL') {
      combinedFactRepresentation += 'Use this section for tone, mood, historical framing, comparisons across eras, why something matters in the broader category, and subjective takes. Do NOT pull numbers, dates, recent claims, or quotes from here. Stats that conflicted with TIER 1A/1B have been replaced with [STAT].\n';
      combinedFactRepresentation += 'This should inform AT MOST 35% of your contextual material. The bulk of every slide comes from TIER 1A/1B facts.\n\n';
    } else {
      combinedFactRepresentation += 'For items not covered by TIER 1B, you may use facts from this section. Stats that conflicted with TIER 1B have been replaced with [STAT].\n\n';
    }

    combinedFactRepresentation += sanitizedPerplexity + '\n\n';
  }

  const sourceList = citations.map((url, i) => `[${i + 1}] ${url}`).join('\n');

  // Raw source excerpt for subjective voice/narrative context (2000-word cap)
  const rawSourceExcerpt = primaryMarkdown
    ? (() => {
        const words = primaryMarkdown.split(/\s+/);
        return words.length > 2000
          ? words.slice(0, 2000).join(' ') + '\n[truncated]'
          : primaryMarkdown;
      })()
    : '';

  console.log(`[mergeSubjectiveResearch] Perplexity sanitization: ${sanitizationStats.statPlaceholders} stats replaced, ${sanitizationStats.quotesRemoved} quotes removed, ${sanitizationStats.entitiesTracked} entities tracked`);

  return {
    ...data,
    perplexityAnswer:    answer,
    combinedFactRepresentation,
    primarySourceUrl,
    citations,
    sourceList,
    researchWordCount: answer.split(/\s+/).length,
    researchOk: answer.length > 500 && citations.length >= 2,
    hasUserSource: !!(userSourceContent || data.factOnlyRepresentation),
    hasUserContext: !!userContextContent,
    sourceQuality,
    alignmentScore: alignScore,
    perplexitySanitizationStats: sanitizationStats,
    rawSourceExcerpt,
  } as unknown as SubjectiveMergedData;
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
Treat it as a reference, not a template. The facts are yours to use; the phrasing is not.
Build each sentence fresh, then verify the facts against TIER 1A afterward.
If TIER 1A provides an order, follow it. If it provides a quote, reproduce it verbatim.
If TIER 1A contradicts anything else, TIER 1A wins. Always.

TIER 1B: USER CONTEXT — WRITER-PROVIDED DATA
The writer pasted this manually. It contains facts, quotes, angles, and instructions.
DATA (quotes, facts, dates, achievements) = use as facts in the relevant slide.
INSTRUCTIONS (tone, emphasis, angles) = follow as directives shaping how you write.
Never quote an instruction as slide content. Never discard TIER 1B data.
If TIER 1B contradicts a specific detail in TIER 1A, TIER 1A wins. Otherwise TIER 1B stands.

TIER 2: PERPLEXITY CONTEXT — SUPPLEMENTARY (35% MAX)
Use for tone calibration, cultural background, and understanding why something matters.
Stats that conflicted with TIER 1A have been replaced with [STAT] — do not invent values for those placeholders.
NEVER use TIER 2 for: specific quotes, dates, stats, achievements, or factual claims WHEN TIER 1A or 1B covers that item.
For items NOT covered by TIER 1A/1B at all, you may use TIER 2 facts — but TIER 2 should inform AT MOST 35% of your contextual material.

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
ORIGINALITY REQUIREMENTS — WRITING FRESH FROM FACTS
═══════════════════════════════════════════════════════════════

You are writing ORIGINAL content using TIER 1A and 1B FACTS. You are NOT paraphrasing.

THE RULE: Facts are yours to use. Language is NOT.
- Use any quote, date, name, achievement from TIER 1A or 1B
- Write every sentence fresh — no copying phrases from sources
- If you find yourself swapping synonyms, you are paraphrasing — rewrite completely

When using a TIER 1A or 1B fact, vary how you present it:
- RESTRUCTURE: if the source uses one long sentence, try two short ones
- ADD A LAYER: pair the fact with your editorial voice — the insight the source does not make

Quick check before moving on: if any 6-word string of your sentence could be Google-searched and land on the source article, rewrite that string.

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
MUST-INCLUDE ITEMS — OVERRIDES TIER 1A ITEM SELECTION
═══════════════════════════════════════════════════════════════

If mandatory items are listed, they define WHICH items appear in the article. This overrides TIER 1A's item list. Even if a mandatory item does not appear in the scraped source, it MUST still get its own dedicated slide.

RULES:
- Every mandatory item = its own slide at full spec. Not a passing mention. Not buried.
- Do NOT substitute a mandatory item with a different item from the scraped source.
- Do NOT drop a mandatory item because it lacks TIER 1A data — use TIER 1B, TIER 2, or TIER 3 for that slide instead.
- TIER 1A still governs facts and ordering for items that DO appear in it.
- After writing, scan the full article and confirm every mandatory item is present. If any is missing, add it and remove a non-mandatory item.

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

Always produce the full slideshow. No exceptions. Never write 'I cannot' or any refusal. If source data is thin, write tighter slides using Tier 2/3. A complete article is always the correct output.${data.userWordCountOverride
    ? `\n\n═══════════════════════════════════════════════════════════════
⚠️ WRITER OVERRIDE — WORD COUNT ⚠️
═══════════════════════════════════════════════════════════════

The writer EXPLICITLY requested ${data.userWordCountOverride.min}-${data.userWordCountOverride.max} words per content slide.
This OVERRIDES the default 35-50 range. ANY slide exceeding ${data.userWordCountOverride.max} words is a FAILURE.
Count every word before outputting each slide. Rewrite if out of range.`
    : ''}`;

  const mandatoryBlock = data.hasMustInclude
    ? `\n\nMANDATORY ITEMS — these override the scraped source's item list (non-negotiable):\n${data.mustIncludeItems.map((item, i) => `${i + 1}. ${item}`).join('\n')}\n\nEvery mandatory item MUST get its own slide, even if it does not appear in TIER 1A. Do NOT substitute any with a different item from the source. Fill remaining slots (up to ${data.slideCount} total) with the best-fit entries.`
    : '';

  const claudeUserPrompt = `SOURCE DATA — write from this:\n\n## STRUCTURED FACT DATABASE\n${data.combinedFactRepresentation}\n\n## RAW SOURCE EXCERPT (narrative voice reference — facts from FACT DATABASE take priority)\n${data.rawSourceExcerpt || '(no raw source available)'}\n\n---\n\nSLIDESHOW ASSIGNMENT:\nTitle: "${data.title}"\nCategory: ${data.category}\nArticle Type: ${data.articleType}\nTone Dial: ${data.toneDial}${data.writingStyle ? '\nStyle Influence: ' + data.writingStyle : ''}\nSlides needed: 1 intro + ${data.slideCount} content slides (MANDATORY — you MUST produce exactly ${data.slideCount} content slides labelled "SLIDE 2" through "SLIDE ${data.slideCount + 1}". No more, no fewer. If the source covers fewer than ${data.slideCount} items, add honorable mentions, related entries, or sister-topic items to reach exactly ${data.slideCount}.)\nSource quality: ${data.sourceQuality}\nPrimary source URL: ${data.primarySourceUrl}${mandatoryBlock}\n\nBEFORE WRITING — checklist:\n1. What is the EXACT promise of the title? (number, emotion, main angle, secondary angle)\n2. Will I produce exactly ${data.slideCount} content slides? (count them before you finish)\n3. What one specific detail from TIER 1A/1B anchors Slide 1?\n4. For ranking articles: am I listing in REVERSE ORDER?\n5. Have I reserved a dedicated slide for every MANDATORY ITEM?\n6. For each slide: what does the source say, and what am I ADDING beyond that?\n7. For any specific date, event, or quote origin: is it confirmed in TIER 1A/1B or am I certain?\n\nWrite the complete slideshow now. Every slide must be labelled "SLIDE N" on its own line — do NOT skip the marker for any slide.`;

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
  const data = await callGrokWithFallback({
    systemContent: systemPrompt,
    userContent:   userPrompt,
    maxTokens:     5000,
    temperature:   0.4,
    timeout:       120_000,
    label:         'generateSubjectiveWithGrok',
  });
  return extractGrokResponseText(data);
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
      else if (slideNum > 1) {
        const minWc = data.userWordCountOverride?.min ?? 35;
        const maxWc = data.userWordCountOverride?.max ?? 50;
        if (wc < minWc || wc > maxWc + 5) warnings.push(`Slide ${slideNum}: ${wc} words (expected ${minWc}-${maxWc})`);
      }
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

  const grokResp = await callGrokWithFallback({
    systemContent,
    userContent,
    maxTokens:   8000,
    temperature: 0.0,
    timeout:     240_000,
    label:       'grokSubjectiveStyleAudit',
  });
  return extractGrokResponseText(grokResp);
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

// ─────────────────────────────────────────────────────────────────────────────
// Slide enrichment — Claude Haiku structured-field extraction for image search
// ─────────────────────────────────────────────────────────────────────────────

interface EnrichedSlideFields {
  primarySubject: string;
  otherSubjects: string[];
  teamName: string;
  eventName: string;
  location: string;
  year: string;
  emotion: string;
}

function emptyEnrichmentFields(): EnrichedSlideFields {
  return {
    primarySubject: '', otherSubjects: [], teamName: '',
    eventName: '', location: '', year: '', emotion: '',
  };
}

const ENRICHER_SYSTEM_PROMPT = `You extract structured fields from MSN slideshow slides for image search. Given a slideshow title, category, and a list of slides (title + body), return ONLY a valid JSON array — no markdown fences, no explanation.

Each element in the array corresponds to one input slide (same order) and must have exactly these fields:

primarySubject (string): FULL first+last name of THE main person for the slide. Real human person ONLY — never a team, school, stadium, trophy, family name, character, or movie title. Spelling MUST match the source exactly. Do not correct, normalize, paraphrase, or invent. Surname-only ('Love', 'Murray') is FORBIDDEN — scan the source for the full first name; if you cannot find it, leave as ''. Position labels ('Running Back Jeremiyah Love', 'Head Coach Andy Reid') are NOT names — extract just the person's name. Leave as '' if the slide is about no specific person.

otherSubjects (string[], max 3): FULL first+last names of OTHER persons relevant to this slide. Same rules as primarySubject. Use for comparisons ('Brady vs Manning' -> primarySubject: 'Tom Brady', otherSubjects: ['Peyton Manning']), trades, multi-person slides. Use [] when no additional persons.

teamName (string): Team or school name only. Examples: 'Detroit Lions', 'Notre Dame', 'Kansas City Chiefs', 'Team Penske', 'Manchester United'. NO sport identifier ('Detroit Lions NFL' is wrong). NO multiple teams ('Lions vs Bears' is wrong). Leave as '' if no team.

eventName (string): Tournament, race, or event MOST RELEVANT TO THIS SPECIFIC SLIDE — not the slideshow umbrella. Examples:
- Slide about "The LIV Golf Complication" inside a Masters slideshow -> eventName: 'LIV Golf' (the slide's topic, NOT 'Masters')
- Slide about "Fowler's Path through the Houston Open" -> eventName: 'Houston Open' (NOT 'Masters')
- Slide about a player's runner-up moment at the 2018 Masters -> eventName: 'Masters' (slide IS about the Masters)
- Slide about "World Ranking Crunch" with no specific event -> eventName: '' (don't default to umbrella event)
Leave as '' if no SLIDE-SPECIFIC event applies. NEVER default to the slideshow-level event just because the slideshow is about it.

location (string): Pure city or geographic location only. Examples: 'Augusta', 'Detroit', 'Las Vegas'. Do NOT put tournaments here (those go in eventName). Leave as '' if none.

year (string): Single 4-digit year. EXTRACT AGGRESSIVELY — if the slide mentions ANY year (game, season, draft, championship, milestone, debut, release), capture the MOST CENTRAL one. Format: 4 digits only ('2024', '1995'). No ranges, no 'season' suffix. Use '' only if NO year appears.

emotion (string): EXACTLY ONE of these canonical values:
- 'celebration' — wins, championships, milestones, records, joyful moments
- 'defeat' — losses, eliminations, blown leads, painful endings
- 'focus' — game action, in-play moments, neutral intensity
- 'concern' — injuries, controversies, struggles, retirement, sad news
- 'confident' — pre-game poses, calm portraits, smiling press shots
- 'intense' — high-stakes plays, rivalries, fights, confrontations
- 'reflection' — retirement tributes, hall-of-fame, historical retrospective
- '' — neutral or no strong tone

Field rules (apply to every slide):
- All fields are INDEPENDENT. Do NOT combine values across fields.
- Do NOT include sport identifiers (NFL, NBA, NASCAR, F1, etc.) in any structured field — sport is auto-detected downstream.
- Do NOT include filler words ('back-to-back', 'record-breaking', 'historic', 'iconic', 'legendary', 'famous') in any structured field.
- teamName vs eventName: NFL/NBA/MLB/NHL/Soccer/Racing crews -> teamName. Tournaments, races, individual-sport events -> eventName.
- eventName vs location: 'Masters' -> eventName, 'Augusta' -> location. 'Wimbledon' -> eventName, 'London' -> location.
- eventName vs slideshow umbrella: eventName is SLIDE-SPECIFIC. Don't paste the slideshow's central event into every slide.

Return ONLY the JSON array. Example: [{"primarySubject":"Tom Brady","otherSubjects":[],"teamName":"New England Patriots","eventName":"Super Bowl","location":"","year":"2019","emotion":"celebration"}]`;

export async function enrichSlides(
  slides: Array<{ title: string; description: string }>,
  slideshowTitle: string,
  slideshowCategory: string,
): Promise<{ slides: EnrichedSlideFields[]; status: string }> {
  if (!slides.length) return { slides: [], status: 'no-slides' };
  if (!ANTHROPIC_KEY) {
    console.warn('[enrichSlides] No ANTHROPIC_API_KEY — returning empty fields');
    return { slides: slides.map(() => emptyEnrichmentFields()), status: 'no-api-key' };
  }

  const slidesText = slides.map((s, i) =>
    `${i + 1}. Title: ${s.title}\n   Body: ${s.description}`
  ).join('\n\n');

  const userPrompt = `Slideshow title: "${slideshowTitle}"\nCategory: ${slideshowCategory}\n\nSlides:\n${slidesText}\n\nReturn the JSON array now.`;

  try {
    const resp = await axios.post(
      'https://api.anthropic.com/v1/messages',
      {
        model: 'claude-haiku-4-5-20251001',
        max_tokens: 3500,
        temperature: 0,
        system: ENRICHER_SYSTEM_PROMPT,
        messages: [{ role: 'user', content: userPrompt }],
      },
      {
        headers: {
          'Content-Type': 'application/json',
          'anthropic-version': '2023-06-01',
          'x-api-key': ANTHROPIC_KEY,
        },
        timeout: 60_000,
      },
    );

    const rawText = ((resp.data?.content as Array<{ text: string }>)?.[0]?.text ?? '').trim();
    const jsonMatch = rawText.match(/\[[\s\S]*\]/);
    if (!jsonMatch) {
      console.warn('[enrichSlides] No JSON array found in Haiku response');
      return { slides: slides.map(() => emptyEnrichmentFields()), status: 'no-json' };
    }

    const parsed = JSON.parse(jsonMatch[0]) as Array<Record<string, unknown>>;
    if (!Array.isArray(parsed)) {
      return { slides: slides.map(() => emptyEnrichmentFields()), status: 'malformed-shape' };
    }

    // Map to typed fields, padding to input length
    const result: EnrichedSlideFields[] = slides.map((_, i) => {
      const raw = parsed[i] ?? {};
      return {
        primarySubject: String(raw.primarySubject ?? ''),
        otherSubjects:  Array.isArray(raw.otherSubjects) ? raw.otherSubjects.map(String).slice(0, 3) : [],
        teamName:       String(raw.teamName ?? ''),
        eventName:      String(raw.eventName ?? ''),
        location:       String(raw.location ?? ''),
        year:           String(raw.year ?? ''),
        emotion:        String(raw.emotion ?? ''),
      };
    });

    console.log(`[enrichSlides] Enriched ${result.length} slides via Claude Haiku`);
    return { slides: result, status: 'success' };
  } catch (err: unknown) {
    if (axios.isAxiosError(err)) {
      console.error(`[enrichSlides] HTTP ${err.response?.status}: ${JSON.stringify(err.response?.data ?? err.message)}`);
    } else {
      console.error('[enrichSlides] Error:', err);
    }
    return { slides: slides.map(() => emptyEnrichmentFields()), status: `error: ${String(err)}` };
  }
}

