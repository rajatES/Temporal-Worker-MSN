import { Worker } from '@temporalio/worker';
import * as activities from './activities';
import * as dotenv from 'dotenv';

dotenv.config();

async function main() {
  const worker = await Worker.create({
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
