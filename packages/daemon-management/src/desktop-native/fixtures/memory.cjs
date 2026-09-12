const { fork } = require('node:child_process');
const { once } = require('node:events');
const { createInterface } = require('node:readline');
const addon = require(process.argv[2]);
let allocation, child;
if (process.argv[3] === 'allocate') {
  allocation = Buffer.alloc(192 * 1024 * 1024, 0x53);
  process.send('ready');
  process.on('disconnect', () => process.exit());
} else {
  const lines = createInterface({ input: process.stdin });
  (async () => {
    for await (const command of lines) {
      try {
        if (command === 'child') {
          child = fork(__filename, [process.argv[2], 'allocate'], { stdio: ['ignore', 'ignore', 'inherit', 'ipc'] });
          await once(child, 'message');
        } else if (command === 'stop') {
          const exited = once(child, 'exit'); child.kill(); await exited; child = undefined;
        } else if (command === 'self') allocation = Buffer.alloc(192 * 1024 * 1024, 0x37);
        console.log(JSON.stringify(await addon.applicationMemory()));
      } catch (error) { console.log(JSON.stringify({ error: error.message })); }
    }
  })().finally(() => { child?.kill(); process.exit(); });
}
