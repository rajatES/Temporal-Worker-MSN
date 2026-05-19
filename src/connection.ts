import { Connection } from '@temporalio/client';
import { NativeConnection } from '@temporalio/worker';

export const temporalAddress   = () => process.env.TEMPORAL_ADDRESS   || 'localhost:7233';
export const temporalNamespace = () => process.env.TEMPORAL_NAMESPACE || 'default';

// When TEMPORAL_API_KEY is set we're talking to Temporal Cloud via API key auth.
// When neither cert nor API key is set we're talking to a local server (no TLS).
function connectionOptions() {
  const apiKey  = process.env.TEMPORAL_API_KEY;
  const address = temporalAddress();

  if (apiKey) {
    return {
      address,
      apiKey,
      tls: true,   // Temporal Cloud always requires TLS; API key replaces the client cert
    };
  }

  // Local Temporal server — no TLS
  return { address };
}

/** For client scripts (trigger, inspect, list, fetch, terminate) */
export async function makeClientConnection(): Promise<Connection> {
  return Connection.connect(connectionOptions());
}

/** For the worker (uses NativeConnection, not Connection) */
export async function makeNativeConnection(): Promise<NativeConnection> {
  return NativeConnection.connect(connectionOptions());
}
