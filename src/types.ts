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

export interface AuditedData extends VerifiedData {
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
}

// ── Subjective pipeline data types ────────────────────────────────────────────

export interface SubjectivePreparedData {
  title: string;
  category: string;
  articleType: string;
  toneDial: string;
  slideCount: number;
  writerName: string;
  writingStyle: string;
  userContext: string;
  userPrimaryUrl: string;
  userSecondaryUrls: string[];           // up to 4 additional URLs (5 total)
  shouldScrapePrimary: boolean;
  hasUserUrl: boolean;
  isPrimaryRestricted: boolean;
  mustIncludeItems: string[];
  hasMustInclude: boolean;
  primaryQuery: string;
  builtInRestricted: string[];
  timestamp: string;
}

export interface SubjectiveResearchedData extends SubjectivePreparedData {
  perplexityAnswer: string;
  citations: string[];
  primaryScraped: string;
  additionalScraped: string[];           // markdown for each userSecondaryUrls entry
}

export interface SubjectiveMergedData extends SubjectiveResearchedData {
  finalContext: string;
  sourceList: string;
  allCitationsCount: number;
  primarySourceUrl: string;
  sourceQuality: 'COMPREHENSIVE' | 'PARTIAL' | 'MINIMAL';
  contextWordCount: number;
  scraped1Ok: boolean;
  scraped2Ok: boolean;
  hasScrapedSource: boolean;
  hasUserContextFlag: boolean;
  researchOk: boolean;
}

export interface SubjectivePromptData extends SubjectiveMergedData {
  claudeSystemPrompt: string;
  claudeUserPrompt: string;
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

// Subjective pipeline is shorter — no fact atomization, no fact verification,
// no citation scraping. Reuses the same IDs the UI's ProgressTracker knows.
export const STAGE_DEFS_SUBJECTIVE: Array<{ id: string; label: string }> = [
  { id: 'parsing',           label: 'Parsing input' },
  { id: 'scraping_source',   label: 'Scraping source' },
  { id: 'researching',       label: 'Researching context' },
  { id: 'building_prompt',   label: 'Merging context' },
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
