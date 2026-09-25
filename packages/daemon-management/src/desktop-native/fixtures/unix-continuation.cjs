const native = require(process.argv[2]);
const fs = require('node:fs');
const path = require('node:path');
const directory = process.argv[3];
const lock = native.acquireLock(path.join(directory, 'owner.lock'));
if (!lock) throw new Error('Cannot acquire fixture ownership');
if (process.argv[4] === 'replacement') {
  const previous = Number(fs.readFileSync(path.join(directory, 'pid'), 'utf8'));
  fs.writeSync(1, JSON.stringify({ pid: process.pid, previous, cwd: process.cwd(), value: process.env.CONTINUATION_FIXTURE,
    argument: process.argv[5], stdin: fs.readFileSync(0, 'utf8') }));
  fs.writeSync(2, 'replacement stderr');
  native.releaseLock(lock);
  process.exit(0);
}
fs.writeFileSync(path.join(directory, 'pid'), String(process.pid));
native.replaceProcess(process.execPath, [__filename, process.argv[2], directory, 'replacement', process.argv[4]],
  Object.entries(process.env).map(([key, value]) => `${key}=${value}`));
throw new Error('Replacement unexpectedly returned');
