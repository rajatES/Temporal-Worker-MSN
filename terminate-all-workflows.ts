import { Client } from '@temporalio/client';
import * as dotenv from 'dotenv';
import { makeClientConnection, temporalNamespace } from './src/connection';

dotenv.config();

(async () => {
  const conn = await makeClientConnection();
  const client = new Client({ connection: conn, namespace: temporalNamespace() });
  
  let count = 0;
  try {
    for await (const wf of client.workflow.list()) {
      try {
        await client.workflow.getHandle(wf.workflowId).terminate('Terminated: bulk termination');
        console.log(`Terminated: ${wf.workflowId}`);
        count++;
      } catch (err) {
        console.log(`Failed to terminate ${wf.workflowId}:`, (err as Error).message);
      }
    }
    console.log(`\nTotal workflows terminated: ${count}`);
  } catch (err) {
    console.error('Error listing workflows:', err);
  }
})().catch(console.error);
