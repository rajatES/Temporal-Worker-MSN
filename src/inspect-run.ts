import { Client } from '@temporalio/client';
import * as dotenv from 'dotenv';
import { makeClientConnection, temporalNamespace } from './connection';
dotenv.config();

async function main() {
  const runId = process.argv[2];
  if (!runId) { console.log('Usage: npx ts-node src/inspect-run.ts <runId>'); process.exit(1); }

  const conn = await makeClientConnection();
  const client = new Client({ connection: conn, namespace: temporalNamespace() });

  let workflowId = '';
  for await (const wf of client.workflow.list({ query: `RunId = '${runId}'` })) {
    workflowId = wf.workflowId;
  }

  const handle = client.workflow.getHandle(workflowId, runId);
  const history = await handle.fetchHistory();
  const events = history.events || [];

  // Find each activity's result
  const activityNames = ['validateStructure', 'extractAuditResults', 'finalAssembly'];
  
  for (const actName of activityNames) {
    const sched = events.find(e => e.activityTaskScheduledEventAttributes?.activityType?.name === actName);
    if (!sched) { console.log(`\n${actName}: NOT FOUND`); continue; }
    const started = events.find(e => e.activityTaskStartedEventAttributes?.scheduledEventId?.toString() === sched.eventId?.toString());
    const completed = events.find(e => e.activityTaskCompletedEventAttributes?.startedEventId?.toString() === started?.eventId?.toString());
    const data = completed?.activityTaskCompletedEventAttributes?.result?.payloads?.[0]?.data;
    if (!data) { console.log(`\n${actName}: NO RESULT`); continue; }
    
    const raw = JSON.parse(Buffer.from(data).toString('utf8'));
    
    if (actName === 'validateStructure') {
      console.log(`\n=== validateStructure ===`);
      console.log(`slides parsed: ${raw.slides?.length ?? 0}`);
      console.log(`articleText first 300 chars:\n${raw.articleText?.slice(0, 300)}`);
    }
    if (actName === 'extractAuditResults') {
      console.log(`\n=== extractAuditResults ===`);
      console.log(`slides in data: ${raw.slides?.length ?? 0}`);
      console.log(`rewriteApplied: ${raw.rewriteApplied}`);
      console.log(`articleText first 300 chars:\n${raw.articleText?.slice(0, 300)}`);
    }
    if (actName === 'finalAssembly') {
      console.log(`\n=== finalAssembly ===`);
      console.log(`articleText first 300 chars:\n${raw.articleText?.slice(0, 300)}`);
    }
  }
}
main().catch(console.error);
