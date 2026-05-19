import { Client } from '@temporalio/client';
import * as dotenv from 'dotenv';
import { makeClientConnection, temporalNamespace } from './src/connection';

dotenv.config();

(async () => {
  const conn = await makeClientConnection();
  const client = new Client({ connection: conn, namespace: temporalNamespace() });
  let wfId;
  for await (const wf of client.workflow.list({ query: "RunId = '019deb01-9613-78d9-b59c-e2000acecec7'" })) {
    wfId = wf.workflowId;
  }
  if (!wfId) { console.log('Not found'); return; }
  await client.workflow.getHandle(wfId).terminate('Terminated: code updated, old history incompatible');
  console.log('Terminated:', wfId);
})().catch(console.error);
