const native = require(process.argv[2]);
const fs = require('node:fs');
const path = require('node:path');
const directory = process.argv[3];
const bundle = path.join(directory, 'Magnitude.app');
if (process.argv[4] === 'installer') {
  const descriptor = Number(process.env.MAGNITUDE_MAC_UPDATE_LEASE_FD);
  const retained = native.adoptMacUpdateLease(bundle, descriptor);
  delete process.env.MAGNITUDE_MAC_UPDATE_LEASE_FD;
  native.validateMacUpdateLease(retained);
  if (native.acquireMacUpdateLease(bundle, false) !== null) throw new Error('Shared admission opened during handoff');
  fs.writeFileSync(path.join(directory, 'installer-pid'), String(process.pid));
  native.replaceProcess(process.execPath, [__filename, process.argv[2], directory, 'replacement'],
    Object.entries(process.env).map(([key, value]) => `${key}=${value}`));
}
if (process.argv[4] === 'replacement') {
  const owner = native.acquireMacUpdateLease(bundle, false);
  if (!owner) throw new Error('Installer admission leaked into replacement');
  fs.writeSync(1, JSON.stringify({ original: Number(fs.readFileSync(path.join(directory, 'pid'), 'utf8')),
    installer: Number(fs.readFileSync(path.join(directory, 'installer-pid'), 'utf8')), replacement: process.pid }));
  native.releaseMacUpdateLease(owner);
  process.exit(0);
}
fs.mkdirSync(bundle);
const retained = native.acquireMacUpdateLease(bundle, true);
if (!retained) throw new Error('Cannot acquire fixture installation');
fs.writeFileSync(path.join(directory, 'pid'), String(process.pid));
const descriptor = native.prepareMacUpdateLeaseExec(retained);
native.replaceProcess(process.execPath, [__filename, process.argv[2], directory, 'installer'],
  Object.entries({ ...process.env, MAGNITUDE_MAC_UPDATE_LEASE_FD: String(descriptor) }).map(([key, value]) => `${key}=${value}`));
