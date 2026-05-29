// ── Input ────────────────────────────────────────────────────────────────────

export type ArticleMode = 'objective' | 'subjective';

export interface FormInput {
  writerName: string;
  title: string;
  category: string;
  slideCount: number;
  slidesPerEntityRaw: string;      // "1 (Standard)" | "2 (Deep Dive - 2 slides per item)"
  sourcesRaw: string;              // newline-separated URLs
  mustIncludeRaw: string;
  userContext: string;
  writingStyle: string;
  // Mode + subjective-only fields (optional for back-compat)
  mode?: ArticleMode;              // defaults to 'objective'
  articleType?: string;            // subjective: Quotes / Opinion & Rankings / etc.
  toneDial?: string;               // subjective: Celebratory / Nostalgic / etc.
}

// ── Intermediate data objects ─────────────────────────────────────────────────

export interface TemporalCtx {
  today: string;
  currentYear: number;
  currentMonth: number;
  currentSeason: string;
  lastSeason: string;
  seasonFormat: string;
  dateAnchor: string;
  seasonAnchor: string;
}

export interface FormatConfig {
  slidesPerEntity: number;
  entityCount: number;
  isMultiSlideFormat: boolean;
  continuationStyle: string;
}

export interface TitleAnalysis {
  promisedCount: number;
  isRanking: boolean;
  isListicle: boolean;
  isTimeBased: boolean;
  emotionalPromise: string | null;
  mainAngle: string;
  secondaryAngle: string | null;
  requiresCorrelation: boolean;
}

export interface SourceAnalysis {
  status: string;
  recommendation: string;
  alignmentScore: number;
  estimatedItems?: number;
  factTypeChecks?: Record<string, boolean>;
  /** Combined markdown of all sources (kept for backwards compat + sourceQuality sizing). */
  scrapedContent: string;
  /** Per-source breakdown — used by atomizeFacts to atomize each source independently
   * and then merge facts by item name. All sources treated as EQUAL Tier 1A authority. */
  scrapedSources?: Array<{ url: string; markdown: string }>;
  sourceCount?: number;
  primaryLength?: number;
  secondaryLength?: number;
  researchMode?: string;
}

export interface AtomizedFact {
  itemNumber: number;
  itemName: string;
  facts: Array<{ type: string; value: string; isExactQuote?: boolean }>;
  rawContent: string;
  /**
   * Short non-numeric narrative excerpt (1-2 sentences, ~220 chars).
   * Reference-only context for facts the regex extractors miss
   * (rare positions, unusual phrasing, current affiliations).
   * Claude is instructed NOT to mirror this for phrasing.
   */
  narrativeContext?: string;
}

// ── Accumulated pipeline data ─────────────────────────────────────────────────
// Each stage spreads the previous object and adds new fields, just like n8n.

export interface PreparedData {
  title: string;
  category: string;
  slideCount: number;
  writerName: string;
  userContext: string;
  writingStyle: string;
  userPrimaryUrl: string;
  userSecondaryUrls: string[];
  hasValidUserSource: boolean;
  isUserUrlRestricted: boolean;
  restrictedDomains: string[];
  mustIncludeItems: string[];
  hasMustInclude: boolean;
  temporalContext: TemporalCtx;
  formatConfig: FormatConfig;
  titleAnalysis: TitleAnalysis;
  sourceCount: number;
  timestamp: string;
  isSports: boolean;
  userWordCountOverride?: { min: number; max: number };
}

export interface SourcedData extends PreparedData {
  sourceAnalysis: SourceAnalysis;
  preferredDomains?: string[];
}

export interface AtomizedData extends SourcedData {
  atomizedFacts: AtomizedFact[];
  factOnlyRepresentation: string;
  sourceSignatures: string[];
  atomizationStats: { itemsProcessed: number; totalFacts: number };
}

export interface RelatedEsArticle {
  title: string;
  url: string;
  sport: string;
  pageviews: number;
}

export interface ResearchedData extends AtomizedData {
  perplexityAnswer: string;
  perplexityCitations: string[];
  perplexityWordCount: number;
  needsRetry: boolean;
  retryReason: string | null;
  hadSoftRefusal: boolean;
  relatedEsArticles: RelatedEsArticle[];
}

export interface MergedData extends ResearchedData {
  combinedFactRepresentation: string;
  primarySourceUrl: string;
  citations: string[];
  sourceList: string;
  researchWordCount: number;
  researchOk: boolean;
  hasUserSource: boolean;
  hasUserContext: boolean;
  sourceQuality: string;
  alignmentScore: number;
  /** Diagnostic counters from lightSanitizePerplexity (set by mergeResearch). */
  perplexitySanitizationStats?: {
    originalLength: number;
    sanitizedLength: number;
    statPlaceholders: number;
    quotesRemoved: number;
    entitiesTracked: number;
  };
}

export interface PromptData extends MergedData {
  claudeSystemPrompt: string;
  claudeUserPrompt: string;
  /** The TIER 1A+1B fact-database section of the user prompt (facts + citation
   *  context), reused verbatim by every batch in batched generation. */
  factContextBlock: string;
}

/** One unit of work in batched generation of large articles (>25 slides). */
export interface BatchSpec {
  batchIndex: number;     // 0-based
  totalBatches: number;
  contentStart: number;   // 1-based presentation position of this batch's first content slide
  contentCount: number;   // number of content slides to produce in this batch
  totalContent: number;   // total content slides across the whole article
  isFirst: boolean;       // first batch also emits TITLE + META + intro
}

export interface GeneratedData extends PromptData {
  articleText: string;
  originalArticleText: string;
  generatedBy: string;
  claudeFailed: boolean;
  failureReason: string | null;
}

export interface Slide {
  slideNum: number;
  title: string;
  body: string;
  wordCount: number;
}

export interface StructuralValidation {
  status: string;
  errors: string[];
  warnings: string[];
  autoFixes: string[];
  slideCount: number;
  metaLength: number;
}

export interface PlagiarismCheck {
  score: number;
  matches: string[];
  status: string;
}

export interface FactProvenance {
  rate: number;
  verified: number;
  unverified: number;
  total: number;
  unverifiedExamples: string[];
}

export interface ValidatedData extends GeneratedData {
  structuralValidation: StructuralValidation;
  plagiarismCheck: PlagiarismCheck;
  factProvenance: FactProvenance;
  slides: Slide[];
}

export interface ClaimItem {
  type: string;
  claim: string;
  context: string;
}

export interface ClaimedData extends ValidatedData {
  claimsToVerify: ClaimItem[];
}

export interface VerificationResult {
  claimIndex: number;
  claim: string;
  status: string;
  finding: string;
  source: string;
}

export interface VerifiedData extends ClaimedData {
  perplexityVerification: {
    results: VerificationResult[];
    citations: string[];
    stats: { verified: number; incorrect: number; unverifiable: number; total: number };
    score: number;
  };
}

export interface SourceEntry {
  index: number;
  url: string;
  factsVerified: string;
  verifiedBy: string;
}

// ── MSN moderation flagging ───────────────────────────────────────────────────
// A robust, NON-BLOCKING flagging layer that audits the generated article (title
// + meta + every slide) against the MSN Content Moderator ruleset. Issues are
// surfaced at the end of generation — nothing is auto-censored or aborted.
//
// Two producers:
//   - 'code'   → deterministic moderationScan activity (absolute banned words,
//                excitable typography, literal banned title patterns, ≥450 chars)
//   - 'claude' → the Haiku audit/moderation node (context/subjective rules the
//                code node can't catch: clickbait framing, swimlane, "Major"
//                justification, sensationalism, etc.)
export type ModerationSeverity = 'absolute' | 'fail' | 'review';
export type ModerationVerdict = 'PASS' | 'REVIEW' | 'FAIL';

export interface ModerationFlag {
  source: 'code' | 'claude';
  severity: ModerationSeverity;   // absolute = would-block (e.g. banned word in title), fail = guideline breach, review = soft/contextual
  rule: string;                   // which rule fired, e.g. "§5a Absolute Ban" / "§4f Bait-and-switch"
  zone: string;                   // 'title' | 'meta' | 'slide 5' | 'body'
  excerpt: string;                // the offending text
  detail: string;                 // human-readable explanation
  suggestion?: string;            // optional, informational only (e.g. censored form "S***") — NEVER auto-applied
}

export interface AuditedData extends VerifiedData {
  // NOTE: `grokAudit` is a legacy field name — this audit is now produced by the
  // Claude (Haiku) moderation node, not Grok. Kept for back-compat with
  // finalAssembly / downstream readers.
  grokAudit: {
    status: string;
    rawResponse: string;
    summary: string;
    stats: {
      rulesPassed: string;
      violations: number;
      corrections: string;
      flags: string;
    };
  };
  grokSources: SourceEntry[];
  combinedSourceList: SourceEntry[];
  combinedSourceListText: string;
  rewriteApplied: boolean;
  /** All moderation issues from both the code node and the Claude audit node. */
  moderationFlags: ModerationFlag[];
  /** Overall verdict derived from the highest-severity flag. Informational — never blocks. */
  moderationVerdict: ModerationVerdict;
}

// ── Subjective pipeline data types ────────────────────────────────────────────
// Aligned with objective pipeline: same data-processing backbone
// (alignment → atomization → structured research → merge), subjective-unique
// voice/style/tone preserved in prompt building and generation.

export interface SubjectivePreparedData extends PreparedData {
  articleType: string;
  toneDial: string;
  shouldScrapePrimary: boolean;
  isPrimaryRestricted: boolean;
  primaryQuery: string;
  builtInRestricted: string[];
}

export interface SubjectiveSourcedData extends SubjectivePreparedData {
  sourceAnalysis: SourceAnalysis;
  preferredDomains?: string[];
}

export interface SubjectiveAtomizedData extends SubjectiveSourcedData {
  atomizedFacts: AtomizedFact[];
  factOnlyRepresentation: string;
  sourceSignatures: string[];
  atomizationStats: { itemsProcessed: number; totalFacts: number };
}

export interface SubjectiveResearchedData extends SubjectiveAtomizedData {
  perplexityAnswer: string;
  perplexityCitations: string[];
  perplexityWordCount: number;
  needsRetry: boolean;
  retryReason: string | null;
  hadSoftRefusal: boolean;
}

export interface SubjectiveMergedData extends SubjectiveResearchedData {
  combinedFactRepresentation: string;
  primarySourceUrl: string;
  citations: string[];
  sourceList: string;
  researchWordCount: number;
  researchOk: boolean;
  hasUserSource: boolean;
  hasUserContext: boolean;
  sourceQuality: string;
  alignmentScore: number;
  perplexitySanitizationStats?: {
    originalLength: number;
    sanitizedLength: number;
    statPlaceholders: number;
    quotesRemoved: number;
    entitiesTracked: number;
  };
  rawSourceExcerpt: string;
}

export interface SubjectivePromptData extends SubjectiveMergedData {
  claudeSystemPrompt: string;
  claudeUserPrompt: string;
  /** Fact-database + raw-source section of the user prompt, reused verbatim by
   *  every batch in batched generation of large subjective articles. */
  factContextBlock: string;
}

export interface SubjectiveGeneratedData extends SubjectivePromptData {
  articleText: string;
  originalArticleText: string;
  generatedBy: string;
  claudeFailed: boolean;
  failureReason: string | null;
}

export interface SubjectiveValidatedData extends SubjectiveGeneratedData {
  validationStatus: 'PASSED' | 'WARNINGS' | 'FAILED';
  errors: string[];
  warnings: string[];
  slideResults: Array<{ slide: number; words: number }>;
}

export interface SubjectiveAuditedData extends SubjectiveValidatedData {
  auditReport: string;
  summaryComment: string;
  auditedArticle: string;
  wasAudited: boolean;
  /** MSN moderation flags (claudeModerate + moderationScan), attached by the workflow. */
  moderationFlags?: ModerationFlag[];
  moderationVerdict?: ModerationVerdict;
}

export interface FinalOutput {
  title: string;
  category: string;
  slideCount: number;
  writerName: string;
  articleText: string;
  originalArticleText: string;
  auditReport: string;
  qualityScore: number;
  researchScore: number;
  verificationScore: number;
  structuralScore: number;
  originalityScore: number;
  primarySourceUrl: string;
  combinedSourceListText: string;
  summaryComment: string;
  validationStatus: string;
  factsVerified: string;
  grokRulesPassed: string;
  flagsForReview: string;
  generatedBy: string;
  generatedAt: string;
  /** Structured MSN moderation flags (code node + Claude audit), merged + deduped. */
  moderationFlags: ModerationFlag[];
  /** Overall moderation verdict (PASS/REVIEW/FAIL) — informational, never blocks. */
  moderationVerdict: ModerationVerdict;
}

export interface SlideshowSlide {
  title: string;
  description: string;
  imageSearch: string;
  primarySubject?: string;
  otherSubjects?: string[];
  teamName?: string;
  eventName?: string;
  location?: string;
  year?: string;
  emotion?: string;
}

export interface WorkflowResult {
  // Matches the shape handleAIGenerate() expects
  title: string;
  metaDescription: string;
  introSlide: { title: string; body: string } | null;
  description: string;
  keywords: string;
  slides: SlideshowSlide[];
  author: string;
  // Quality summary for the UI
  qualityScore: number;
  summaryComment: string;
  flagsForReview: string;
  generatedBy: string;
  enrichmentStatus?: string;
  // MSN moderation flagging (non-blocking) — surfaced in the generator UI.
  moderationFlags?: ModerationFlag[];
  moderationVerdict?: ModerationVerdict;
}

// ── Progress tracking ─────────────────────────────────────────────────────────

export type StageStatus = 'pending' | 'active' | 'complete' | 'warning' | 'error' | 'awaiting_human';

export interface Stage {
  id: string;
  label: string;
  status: StageStatus;
  detail?: string;
}

export const STAGE_DEFS: Array<{ id: string; label: string }> = [
  { id: 'parsing',           label: 'Parsing input' },
  { id: 'scraping_source',   label: 'Scraping source' },
  { id: 'analyzing',         label: 'Analyzing alignment' },
  { id: 'atomizing',         label: 'Extracting facts' },
  { id: 'researching',       label: 'Researching' },
  { id: 'scraping_citations',label: 'Scraping citations' },
  { id: 'building_prompt',   label: 'Building prompt' },
  { id: 'generating',        label: 'Generating article' },
  { id: 'validating',        label: 'Validating structure' },
  { id: 'verifying',         label: 'Verifying facts' },
  { id: 'auditing',          label: 'Auditing with Grok' },
  { id: 'creating_docs',     label: 'Assembling output' },
  { id: 'enriching',         label: 'Enriching slides' },
  { id: 'complete',          label: 'Done' },
];

// Subjective pipeline — aligned with objective data-processing backbone,
// but keeps subjective voice/style and skips fact verification + human pauses.
export const STAGE_DEFS_SUBJECTIVE: Array<{ id: string; label: string }> = [
  { id: 'parsing',           label: 'Parsing input' },
  { id: 'scraping_source',   label: 'Scraping source' },
  { id: 'analyzing',         label: 'Analyzing alignment' },
  { id: 'atomizing',         label: 'Extracting facts' },
  { id: 'researching',       label: 'Researching context' },
  { id: 'scraping_citations',label: 'Scraping citations' },
  { id: 'building_prompt',   label: 'Building prompt' },
  { id: 'generating',        label: 'Generating article' },
  { id: 'validating',        label: 'Validating structure' },
  { id: 'auditing',          label: 'Style audit (Grok)' },
  { id: 'creating_docs',     label: 'Assembling output' },
  { id: 'enriching',         label: 'Enriching slides' },
  { id: 'complete',          label: 'Done' },
];

// ── Human-in-the-loop ─────────────────────────────────────────────────────────

export interface HumanReviewOption {
  label: string;
  value: string;
  requiresInput?: boolean;   // if true, UI shows a textarea
  inputPlaceholder?: string;
}

export interface HumanReviewRequest {
  id: string;
  type: 'poor_source' | 'thin_research' | 'validation_errors' | 'fact_errors' | 'low_quality';
  message: string;
  details: string[];
  options: HumanReviewOption[];
}

export interface HumanDecision {
  requestId: string;
  choice: string;
  additionalInput?: string;
}

export interface WorkflowProgress {
  stages: Stage[];
  currentStageId: string;
  humanReviewRequest: HumanReviewRequest | null;
  isComplete: boolean;
  result?: WorkflowResult;
  error?: string;
}
