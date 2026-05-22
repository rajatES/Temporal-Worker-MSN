import { Client } from '@temporalio/client';
import * as dotenv from 'dotenv';
import { makeClientConnection, temporalNamespace } from './connection';

dotenv.config();

function parseSlides(articleText: string, category: string) {
  const slides: { slideNum: number; title: string; body: string }[] = [];
  let cur: number | null = null, title = '', body = '';
  for (const line of articleText.split('\n')) {
    const t = line.trim();
    const m = t.match(/^SLIDE\s*(\d+)/i);
    if (m) {
      if (cur !== null) slides.push({ slideNum: cur, title, body: body.trim() });
      cur = parseInt(m[1]); title = ''; body = '';
    } else if (cur !== null && !title && t && !t.startsWith('META:')) {
      title = t;
    } else if (cur !== null && title && t && !t.startsWith('SOURCES')) {
      body += (body ? ' ' : '') + t;
    }
  }
  if (cur !== null) slides.push({ slideNum: cur, title, body: body.trim() });
  const intro   = slides.find(s => s.slideNum === 1);
  const content = slides
    .filter(s => s.slideNum > 1)
    .sort((a, b) => a.slideNum - b.slideNum)
    .map(s => ({
      title: s.title, description: s.body, imageSearch: `${s.title} ${category}`,
    }));
  return { intro, content };
}

async function extractFromHistory(client: Client, workflowId: string, runId?: string) {
  const handle  = client.workflow.getHandle(workflowId, runId);
  const history = await handle.fetchHistory();
  const events  = history.events || [];

  const sched     = events.find(e => e.activityTaskScheduledEventAttributes?.activityType?.name === 'finalAssembly');
  const started   = events.find(e => e.activityTaskStartedEventAttributes?.scheduledEventId?.toString() === sched?.eventId?.toString());
  const completed = events.find(e => e.activityTaskCompletedEventAttributes?.startedEventId?.toString() === started?.eventId?.toString());
  const data      = completed?.activityTaskCompletedEventAttributes?.result?.payloads?.[0]?.data as Buffer | undefined;
  if (!data) throw new Error('finalAssembly activity result not found in workflow history');

  const raw = JSON.parse(data.toString('utf8'));

  // New format: already has slides[]
  if (Array.isArray(raw.slides)) return raw;

  // Old format: has articleText — parse it
  const { intro, content } = parseSlides(raw.articleText || '', raw.category || '');
  return {
    title:          raw.title       ?? '',
    description:    intro?.body     ?? '',
    keywords:       raw.category    ?? '',
    author:         raw.writerName  ?? '',
    slides:         content,
    qualityScore:   raw.qualityScore,
    generatedBy:    raw.generatedBy,
    summaryComment: raw.summaryComment,
    flagsForReview: raw.flagsForReview,
  };
}

async function main() {
  const runId = process.argv[2];
  if (!runId) {
    console.error('Usage: npx ts-node src/fetch-result.ts <runId>');
    process.exit(1);
  }

  const connection = await makeClientConnection();
  const client     = new Client({ connection, namespace: temporalNamespace() });

  // Resolve runId → workflowId
  let workflowId: string | undefined;
  for await (const wf of client.workflow.list({ query: `RunId = '${runId}'` })) {
    workflowId = wf.workflowId;
  }
  if (!workflowId) throw new Error(`No workflow found for runId: ${runId}`);

  console.error(`Workflow: ${workflowId} (${runId})`);

  let result: any;
  try {
    const handle = client.workflow.getHandle(workflowId, runId);
    result = await handle.result();
    // New format already has slides[]
    if (!Array.isArray(result.slides)) {
      const { intro, content } = parseSlides(result.articleText || '', result.category || '');
      result = { title: result.title, description: intro?.body || '', keywords: result.category, author: result.writerName, slides: content, qualityScore: result.qualityScore, generatedBy: result.generatedBy, summaryComment: result.summaryComment, flagsForReview: result.flagsForReview };
    }
  } catch {
    // Workflow failed — recover from history
    console.error('Workflow failed — recovering from activity history…');
    result = await extractFromHistory(client, workflowId, runId);
  }

  console.log(JSON.stringify(result, null, 2));
}

main().catch(err => { console.error(err.message); process.exit(1); });
