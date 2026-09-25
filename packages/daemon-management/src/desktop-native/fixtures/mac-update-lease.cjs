const native = require(process.argv[2])
const lease = native.acquireMacUpdateLease(process.argv[3], process.argv[4] === 'Exclusive')
process.stdout.write(lease === null ? 'busy\n' : 'ready\n')
if (lease !== null) {
  process.stdin.resume()
  process.stdin.on('end', () => { native.releaseMacUpdateLease(lease); process.exit(0) })
}
