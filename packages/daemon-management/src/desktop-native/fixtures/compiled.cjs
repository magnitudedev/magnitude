// Static require is intentional: Bun must embed the native addon in its executable.
const native = require('../../../dist/native/' + process.platform + '-' + process.arch + '/desktop-host.node');
const { spawn } = require('node:child_process');
native.guardParent(0);
const worker = spawn('/bin/sleep', ['60'], { stdio: 'ignore' });
worker.once('spawn', () => {
  console.log(JSON.stringify({ child: process.pid, worker: worker.pid }));
  for (;;) {}
});
