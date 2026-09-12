const assert = require('node:assert/strict');
const { randomUUID } = require('node:crypto');
const net = require('node:net');
assert.equal(process.platform, 'win32', 'This is native Windows acceptance, not a simulation.');
if (process.argv.includes('--stdio-fixture-child')) {
  const ownerNative = require(process.env.MAGNITUDE_FIXTURE_NATIVE_ADDON);
  ownerNative.guardParent(0);
  process.stdin.setEncoding('utf8');
  let input = '';
  process.stdin.on('data', chunk => {
    input += chunk;
    if (input.includes('\n')) { process.stdout.write('DATA ' + input); process.stderr.write('DIAGNOSTIC\n'); input = ''; }
  });
  process.stdin.on('end', () => process.stdout.write('EOF\n', () => process.exit(0)));
} else if (process.argv.includes('--owned-fixture-child')) {
  const ownerNative = require(process.env.MAGNITUDE_FIXTURE_NATIVE_ADDON);
  ownerNative.guardParent(0);
  assert.throws(() => ownerNative.guardParent(0), /already validated/);
  process.stdout.write(JSON.stringify({ pid: process.pid, argument: process.argv.at(-1), empty: process.env.MAGNITUDE_FIXTURE_EMPTY, unicode: process.env.MAGNITUDE_FIXTURE_UNICODE }) + '\n');
  setInterval(() => {}, 1000);
} else {
const native = require(process.argv.at(-1));
const pipes = new Set();
const clients = new Set();
const locks = new Set();
const observers = new Set();
const fs = require('node:fs');
const path = require('node:path');
const localAppData = native.localAppDataDirectory();
assert.equal(typeof localAppData, 'string');
assert.ok(path.win32.isAbsolute(localAppData));
const inheritedLocalAppData = process.env.LOCALAPPDATA;
try {
  process.env.LOCALAPPDATA = '\\\\not-a-local-profile\\ignored';
  assert.equal(native.localAppDataDirectory(), localAppData, 'Known-folder lookup must not derive ownership from the caller environment');
} finally {
  if (inheritedLocalAppData === undefined) delete process.env.LOCALAPPDATA;
  else process.env.LOCALAPPDATA = inheritedLocalAppData;
}
const lockDirectory = fs.mkdtempSync(path.join(require('node:os').tmpdir(), 'magnitude-lock-fixture-'));
const deadline = setTimeout(() => { console.error('Windows native pipe fixture timed out'); process.exit(1); }, 20000);
const create = () => {
  const name = `\\\\.\\pipe\\magnitude-fixture-${randomUUID()}`;
  const pipe = native.createPrivatePipe(name, true);
  pipes.add(pipe);
  return { name, pipe };
};
const connected = async () => {
  const { name, pipe } = create();
  const accepted = native.acceptPrivatePipe(pipe);
  const client = new net.Socket();
  clients.add(client);
  client.on('error', () => {});
  await new Promise((resolve, reject) => { client.once('error', reject); client.connect(name, resolve); });
  assert.equal(await accepted, process.pid);
  return { pipe, client };
};
const readExactly = async (pipe, size) => {
  const chunks = []; let length = 0;
  while (length < size) {
    const chunk = await native.readPrivatePipe(pipe);
    assert.ok(chunk.length, 'No early EOF');
    chunks.push(chunk); length += chunk.length;
  }
  return Buffer.concat(chunks);
};
(async () => {
  try {
    const lockPath = path.join(lockDirectory, 'private', 'application.lock');
    assert.equal(native.inspectApplicationEndpoint(path.dirname(lockPath)), null);
    assert.equal(fs.existsSync(path.dirname(lockPath)), false);
    assert.throws(() => native.acquireLock(lockPath + '\0ignored'), /NUL/);
    assert.equal(fs.existsSync(lockPath), false);
    assert.throws(() => native.acquireLock('relative.lock'), /absolute local drive/);
    const lock = native.acquireLock(lockPath); locks.add(lock);
    const endpoint = native.lockEndpoint(lock);
    assert.ok(endpoint.startsWith('\\\\.\\pipe\\magnitude-app-'));
    assert.equal(native.inspectApplicationEndpoint(path.dirname(lockPath)), endpoint);
    assert.equal(native.inspectApplicationEndpoint('\\\\?\\' + path.dirname(lockPath)), endpoint);
    assert.equal(native.inspectApplicationEndpoint(path.join(lockDirectory, 'PRIVATE')), endpoint);
    assert.equal(native.acquireLock(lockPath), null);
    assert.equal(native.acquireLock('\\\\?\\' + lockPath), null);
    assert.throws(() => native.releaseLock({}), /Invalid ownership lock/);
    assert.throws(() => native.releaseLock(Object.create(lock)), /Invalid ownership lock/);
    native.releaseLock(lock); native.releaseLock(lock); locks.delete(lock);
    assert.throws(() => native.lockEndpoint(lock), /live ownership lock/);
    const alias = path.join(path.dirname(lockPath), 'alias.lock'); fs.linkSync(lockPath, alias);
    assert.throws(() => native.acquireLock(alias), /Unsafe or inaccessible/);
    console.log('PASS tagged ownership lock, malformed paths, extended path contention and hard-link rejection');
    assert.throws(() => native.readPrivatePipe({}), /owned native pipe/);
    const { pipe, client } = await connected();
    const payload = Buffer.alloc(65000, 0xa7);
    const received = readExactly(pipe, payload.length);
    client.write(payload);
    assert.deepEqual(await received, payload);
    const reply = new Promise(resolve => client.once('data', resolve));
    assert.equal(await native.writePrivatePipe(pipe, Buffer.from('response')), 8);
    assert.deepEqual(await reply, Buffer.from('response'));
    client.destroy(); clients.delete(client);
    assert.equal((await native.readPrivatePipe(pipe)).length, 0);
    await native.closePrivatePipe(pipe);
    await native.closePrivatePipe(pipe);
    pipes.delete(pipe);
    await assert.rejects(native.readPrivatePipe(pipe), { win32Code: 995 });
    console.log('PASS Node-API duplex bytes, native client identity, EOF and terminal close');

    // Application replies are written before request cleanup closes the server handle.
    // Read only after close to catch loss of buffered replies in each supported JS host.
    const retiring = await connected();
    const finalReply = Buffer.alloc(65000, 0x3d);
    assert.equal(await native.writePrivatePipe(retiring.pipe, finalReply), finalReply.length);
    await native.closePrivatePipe(retiring.pipe); pipes.delete(retiring.pipe);
    const drained = [];
    for await (const chunk of retiring.client) drained.push(chunk);
    assert.deepEqual(Buffer.concat(drained), finalReply);
    clients.delete(retiring.client);
    console.log('PASS buffered reply survives server-handle close before client read');

    // The same entry point works as Node, Bun source, or a compiled Bun executable.
    const output = create();
    const quoted = value => '"' + value.replace(/(\\*)"/g, '$1$1\\"').replace(/(\\+)$/g, '$1$1') + '"';
    const argument = 'spaces "quotes" and trailing slash\\';
    const command = [process.execPath, process.argv[1], '--owned-fixture-child', argument].map(quoted).join(' ');
    const environment = Object.entries({ ...process.env, MAGNITUDE_FIXTURE_EMPTY: "", MAGNITUDE_FIXTURE_UNICODE: "模型🙂", MAGNITUDE_FIXTURE_NATIVE_ADDON: process.argv.at(-1) }).filter(([key, value]) => !key.startsWith('=') && value !== undefined)
      .map(([key, value]) => `${key}=${value}`).join('\0') + '\0\0';
    assert.throws(() => native.ownedProcessIdentity({}), /owned Windows process/);
    assert.throws(() => native.spawnOwnedProcess(process.execPath, command, 'Path=a\0PATH=b\0\0', output.name), { win32Code: 87 });
    assert.throws(() => native.spawnOwnedProcess(process.execPath, command, 'A=a\0\0B=b\0\0', output.name), /Invalid owned Windows process arguments/);
    const owned = native.spawnOwnedProcess(process.execPath, command, environment, output.name);
    try {
      const identity = native.ownedProcessIdentity(owned);
      assert.match(identity.creationTime, /^[0-9a-f]{16}$/);
      const observation = native.observeProcess(identity.pid);
      assert.notEqual(observation, null); observers.add(observation);
      assert.equal(native.observedProcessExited(observation), false);
      const observed = native.observedProcessDetails(observation);
      assert.equal(observed.pid, identity.pid);
      assert.equal(observed.creationTime, identity.creationTime);
      assert.equal(observed.executable.toLowerCase(), process.execPath.toLowerCase());
      assert.match(observed.userSid, /^S-1-/);
      const parentSnapshot = native.snapshotProcessParents();
      assert.equal(new Set(parentSnapshot.map(row => row.pid)).size, parentSnapshot.length);
      assert.deepEqual(parentSnapshot.find(row => row.pid === identity.pid), { pid: identity.pid, parentPid: process.pid });
      assert.throws(() => native.observedProcessDetails(owned), { win32Code: 6 });
      assert.throws(() => native.observedProcessDetails(Object.create(observation)), { win32Code: 6 });
      assert.throws(() => native.terminateOwnedProcess(observation), /owned Windows process/);
      assert.equal(native.ownedProcessExit(owned), null);
      assert.ok(native.ownedProcessActiveCount(owned) >= 1);
      await native.acceptPrivatePipe(output.pipe);
      let text = '';
      while (!text.includes('\n')) text += (await native.readPrivatePipe(output.pipe)).toString();
      assert.deepEqual(JSON.parse(text), { pid: identity.pid, argument, empty: "", unicode: "模型🙂" });
      assert.throws(() => native.startMigrationProcessRetirement(observation), { win32Code: 6 });
      assert.throws(() => native.acquireMigrationProcess(identity.pid, '0000000000000000', observed.executable, observed.userSid), { win32Code: 13 });
      assert.throws(() => native.acquireMigrationProcess(identity.pid, identity.creationTime, observed.executable + '\0suffix', observed.userSid), { win32Code: 87 });
      const migration = native.acquireMigrationProcess(identity.pid, identity.creationTime, observed.executable, observed.userSid);
      try {
        assert.throws(() => native.startMigrationProcessRetirement(Object.create(migration)), { win32Code: 6 });
        assert.equal(native.migrationProcessExited(migration), false);
        native.startMigrationProcessRetirement(migration);
        while (!native.migrationProcessExited(migration)) await new Promise(resolve => setTimeout(resolve, 10));
        native.startMigrationProcessRetirement(migration);
      } finally { native.releaseMigrationProcess(migration); }
      native.releaseMigrationProcess(migration);
      assert.throws(() => native.startMigrationProcessRetirement(migration), { win32Code: 6 });
      console.log('PASS exact migration capability, identity mismatch, retained exit and terminal close');

      while (native.ownedProcessActiveCount(owned) !== 0) await new Promise(resolve => setTimeout(resolve, 10));
      assert.equal(native.ownedProcessExit(owned), 1);
      assert.deepEqual(native.ownedProcessIdentity(owned), identity);
      assert.equal(native.observedProcessExited(observation), true);
      native.releaseObservedProcess(observation); native.releaseObservedProcess(observation);
      observers.delete(observation);
      assert.throws(() => native.observedProcessExited(observation), { win32Code: 6 });
      assert.throws(() => native.observedProcessDetails(observation), { win32Code: 6 });
    } finally {
      native.closeOwnedProcess(owned);
      native.closeOwnedProcess(owned);
      await native.closePrivatePipe(output.pipe); pipes.delete(output.pipe);
    }
    assert.throws(() => native.ownedProcessActiveCount(owned), { win32Code: 6 });
    console.log('PASS Node-API atomic job ownership, exact identity, inherited output, quoted argv, retirement and idempotent release');

    const inputStream = create(), outputStream = create(), errorStream = create();
    const streamCommand = [process.execPath, process.argv[1], '--stdio-fixture-child'].map(quoted).join(' ');
    assert.throws(() => native.spawnOwnedProcessWithPipes(process.execPath, streamCommand, environment, inputStream.name, inputStream.name, errorStream.name), /Invalid owned Windows process arguments/);
    const streamJob = native.spawnOwnedProcessWithPipes(process.execPath, streamCommand, environment, inputStream.name, outputStream.name, errorStream.name);
    try {
      await Promise.all([inputStream, outputStream, errorStream].map(stream => native.acceptPrivatePipe(stream.pipe)));
      const line = async pipe => {
        let text = '';
        while (!text.includes('\n')) {
          const chunk = await native.readPrivatePipe(pipe);
          assert.ok(chunk.length > 0, 'Stream ended before its record'); text += chunk.toString();
        }
        return text;
      };
      const payload = Buffer.from('parent-lifetime\n');
      for (let offset = 0; offset < payload.length;) offset += await native.writePrivatePipe(inputStream.pipe, payload.subarray(offset));
      assert.equal(await line(outputStream.pipe), 'DATA parent-lifetime\n');
      assert.equal(await line(errorStream.pipe), 'DIAGNOSTIC\n');
      await native.closePrivatePipe(inputStream.pipe); pipes.delete(inputStream.pipe);
      assert.equal(await line(outputStream.pipe), 'EOF\n');
      while (native.ownedProcessExit(streamJob) === null) await new Promise(resolve => setTimeout(resolve, 10));
      assert.equal(native.ownedProcessExit(streamJob), 0);
      while (native.ownedProcessActiveCount(streamJob) !== 0) await new Promise(resolve => setTimeout(resolve, 10));
    } finally {
      native.closeOwnedProcess(streamJob);
      await Promise.all([inputStream, outputStream, errorStream].map(async stream => { await native.closePrivatePipe(stream.pipe); pipes.delete(stream.pipe); }));
    }
    console.log('PASS separate inherited stdin/stdout/stderr, parent EOF and job retirement');

    const taskOutput = create();
    const queryPath = path.join(path.dirname(process.argv.at(-1)), 'magnitude-task-query.exe');
    const queryAccepted = native.acceptPrivatePipe(taskOutput.pipe);
    const queryJob = native.spawnOwnedProcess(queryPath, quoted(queryPath), environment, taskOutput.name);
    let queryDeadline;
    try {
      await Promise.race([(async () => {
        assert.equal(await queryAccepted, process.pid, 'Only the parent opens the inherited output handle');
        const chunks = []; let size = 0;
        for (;;) {
          const chunk = await native.readPrivatePipe(taskOutput.pipe);
          if (!chunk.length) break;
          size += chunk.length; assert.ok(size <= 512 * 1024, 'Task query output is bounded');
          chunks.push(chunk);
        }
        const result = JSON.parse(Buffer.concat(chunks).toString('utf8'));
        assert.ok(['Missing', 'Registered'].includes(result._tag), JSON.stringify(result));
        if (result._tag === 'Registered') {
          assert.match(result.currentUserSid, /^S-1-/);
          assert.ok(result.xml.includes('<Task'));
        }
        while (native.ownedProcessExit(queryJob) === null || native.ownedProcessActiveCount(queryJob) !== 0)
          await new Promise(resolve => setTimeout(resolve, 10));
        assert.equal(native.ownedProcessExit(queryJob), 0);
      })(), new Promise((_, reject) => { queryDeadline = setTimeout(() => reject(new Error('Task query deadline')), 5000); })]);
    } finally {
      clearTimeout(queryDeadline);
      native.terminateOwnedProcess(queryJob);
      native.closeOwnedProcess(queryJob);
      await native.closePrivatePipe(taskOutput.pipe); pipes.delete(taskOutput.pipe);
    }
    console.log('PASS contained read-only Task Scheduler inspection');

    // More idle reads than Node's default worker pool must not starve write/close work.
    const peers = await Promise.all(Array.from({ length: 24 }, connected));
    const incoming = peers.map(({ pipe }) => native.readPrivatePipe(pipe));
    const outgoing = peers.map(({ client }) => new Promise(resolve => client.once('data', resolve)));
    assert.deepEqual(await Promise.all(peers.map(({ pipe }) => native.writePrivatePipe(pipe, Buffer.from('out')))), Array(24).fill(3));
    assert.ok((await Promise.all(outgoing)).every(value => value.equals(Buffer.from('out'))));
    peers.forEach(({ client }) => client.write('in'));
    assert.ok((await Promise.all(incoming)).every(value => value.equals(Buffer.from('in'))));
    const pending = peers.map(({ pipe }) => native.readPrivatePipe(pipe).then(() => assert.fail('Read should be cancelled'), error => assert.equal(error.win32Code, 995)));
    await Promise.all(peers.map(({ pipe }) => native.closePrivatePipe(pipe)));
    await Promise.all(pending);
    peers.forEach(({ pipe, client }) => { pipes.delete(pipe); client.destroy(); clients.delete(client); });
    const listening = create();
    const cancelledAccept = native.acceptPrivatePipe(listening.pipe).then(() => assert.fail('Accept should be cancelled'), error => assert.equal(error.win32Code, 995));
    await native.closePrivatePipe(listening.pipe); await cancelledAccept; pipes.delete(listening.pipe);
    console.log('PASS 24 concurrent pipes without worker starvation and pending read/accept cancellation');
  } finally {
    for (const client of clients) client.destroy();
    await Promise.all([...pipes].map(pipe => native.closePrivatePipe(pipe)));
    for (const observation of observers) native.releaseObservedProcess(observation);
    for (const lock of locks) native.releaseLock(lock);
    fs.rmSync(lockDirectory, { recursive: true, force: true });
    clearTimeout(deadline);
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

}
