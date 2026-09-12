const { app } = require('electron');
const native = require(process.argv[2]);
let lock;
app.whenReady().then(() => {
  lock = native.acquireLock(process.argv[3]);
  if (!lock) throw new Error('Electron lock was unexpectedly contended');
  if (native.acquireLock(process.argv[3]) !== null) throw new Error('Electron acquired a second lock');
  native.releaseLock(lock);
  native.releaseLock(lock);
  lock = native.acquireLock(process.argv[3]);
  if (!lock) throw new Error('Electron lock was not released');
  native.releaseLock(lock);
  console.log('electron-native-pass');
  app.quit();
}).catch(error => { console.error(error); app.exit(1); });
