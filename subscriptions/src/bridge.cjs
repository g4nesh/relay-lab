'use strict';
// Only fixed, payload-free invalidations leave this trusted process. SDK errors
// can contain URLs/tokens: never forward console output or exception messages.
for (const method of ['log', 'info', 'warn', 'error', 'debug']) console[method] = () => {};
const { DbConnection } = require('../dist/module_bindings/index.js');
const { Invalidations } = require('./invalidations.cjs');
const output = new Invalidations(line => {
  if (process.stdout.writableLength > 8192) return process.exit(1);
  process.stdout.write(line);
});
let connection;
let stopping = false;
const fail = () => {
  if (stopping) return;
  stopping = true;
  output.status('degraded');
  setTimeout(() => process.exit(1), 10);
};
process.on('uncaughtException', fail);
process.on('unhandledRejection', fail);
process.on('SIGTERM', () => { stopping = true; connection?.disconnect(); process.exit(0); });
process.stdout.on('error', () => process.exit(1));
let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', part => {
  input += part;
  if (input.length > 32768) fail();
});
process.stdin.on('end', () => {
  try {
    const config = JSON.parse(input);
    input = '';
    if (typeof config.token !== 'string' || !config.token ||
        typeof config.server !== 'string' || typeof config.database !== 'string') return fail();
    const watches = [
      ['factoryRun', 'factory_run', 'tasks'],
      ['factoryEmbedding', 'factory_embedding', 'coordination'],
      ['factorySemanticPacket', 'factory_semantic_packet', 'coordination'],
      ['studioPresence', 'studio_presence', 'workers'],
      ['studioTaskWatch', 'studio_task_watch', 'tasks'],
      ['studioAttention', 'studio_attention', 'coordination'],
      ['studioJob', 'studio_job', 'coordination'],
      ['studioRollup', 'studio_rollup', 'coordination'],
    ];
    const handshake = setTimeout(fail, 15000);
    connection = DbConnection.builder()
      .withUri(config.server).withDatabaseName(config.database).withToken(config.token)
      .withCompression('none')
      .onConnect(conn => {
        for (const [accessor, , topic] of watches) {
          const table = conn.db[accessor];
          if (!table) return fail(); // A mismatched module/binding is not a healthy feed.
          table.onInsert(() => output.add(topic));
          table.onDelete(() => output.add(topic));
          table.onUpdate(() => output.add(topic));
        }
        conn.subscriptionBuilder()
          .onApplied(() => { clearTimeout(handshake); output.status('ready'); })
          .onError(fail)
          .subscribe(watches.map(([, table]) => `SELECT * FROM ${table}`));
      }).onConnectError(fail).onDisconnect(fail).build();
    // Silence is distinguishable from a wedged subscription process.
    setInterval(() => output.heartbeat(), 15000);
  } catch { fail(); }
});
