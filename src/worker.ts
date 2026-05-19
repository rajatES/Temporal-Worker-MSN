import { Worker } from '@temporalio/worker';
import * as activities from './activities';
import * as dotenv from 'dotenv';
import { makeNativeConnection, temporalAddress, temporalNamespace } from './connection';

dotenv.config();

async function main() {
  const connection = await makeNativeConnection();
  console.log(`Connecting to Temporal at ${temporalAddress()} (namespace: ${temporalNamespace()})`);

  const worker = await Worker.create({
    connection,
    namespace: temporalNamespace(),
    workflowsPath: require.resolve('./workflows'),
    activities,
    taskQueue: 'msn-article-generator',
  });

  console.log('Worker started on task queue: msn-article-generator');
  await worker.run();
}

main().catch(err => {
  console.error('Worker failed:', err);
  process.exit(1);
});
