import { Client } from '@temporalio/client';
import * as dotenv from 'dotenv';
import { makeClientConnection, temporalNamespace } from './connection';
dotenv.config();

async function main() {
  const conn = await makeClientConnection();
  const client = new Client({ connection: conn, namespace: temporalNamespace() });

  console.log('Recent workflows:\n');
  let count = 0;
  for await (const wf of client.workflow.list({ pageSize: 10 })) {
    console.log(`runId:      ${wf.runId}`);
    console.log(`status:     ${wf.status.name}`);
    console.log(`start:      ${wf.startTime}`);
    console.log(`close:      ${wf.closeTime ?? 'running'}`);
    console.log('---');
    if (++count >= 10) break;
  }
}
main().catch(console.error);
