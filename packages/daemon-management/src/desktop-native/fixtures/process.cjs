// Real-process acceptance fixture, deliberately independent of the app runtime.
const { spawn } = require('node:child_process');
const native = require(process.argv[2]);
const mode = process.argv[3];
if (mode === 'owner') {
  const lock = native.acquireLock(process.argv[4]);
  console.log(lock ? 'owned' : 'contended');
  if (!lock) process.exit(2);
  // Keep a strong reference even when GC runs.
  global.lock = lock;
  setInterval(() => {}, 1000);
} else if (mode === 'owner-child') {
  global.lock = native.acquireLock(process.argv[4]);
  const worker = spawn('/bin/sleep', ['60'], { stdio: 'ignore' });
  console.log(JSON.stringify({ owner: process.pid, worker: worker.pid }));
  setInterval(() => {}, 1000);
} else if (mode === 'owned' || mode === 'owned-stubborn') {
  native.guardParent(0);
  if (mode === 'owned-stubborn') process.on('SIGTERM', () => {});
  const worker = spawn('/bin/sleep', ['60'], { stdio: 'ignore' });
  worker.once('spawn', () => process.stderr.write(JSON.stringify({ child: process.pid, worker: worker.pid }) + '\n'));
  setInterval(() => {}, 1000);
} else if (mode === 'guard') {
  native.guardParent(0);
  const worker = spawn('/bin/sleep', ['60'], { stdio: 'ignore' });
  worker.once('spawn', () => {
    console.log(JSON.stringify({ child: process.pid, worker: worker.pid }));
    if (process.argv[4] === 'blocked') { for (;;) {} }
    setInterval(() => {}, 1000);
  });
} else if (mode === 'parent') {
  const child = spawn(process.execPath, [__filename, process.argv[2], 'guard', process.argv[4]], {
    detached: true, stdio: ['pipe', 'pipe', 'inherit'],
  });
  child.stdout.pipe(process.stdout);
  setInterval(() => {}, 1000);
} else if (mode === 'invalid-guard') {
  try { native.guardParent(1); process.exit(3); }
  catch { console.log('rejected'); }
}
