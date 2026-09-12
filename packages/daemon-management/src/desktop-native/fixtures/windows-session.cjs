const assert = require('node:assert/strict');
const [addon, expected] = process.argv.slice(2);
assert.equal(process.platform, 'win32');
assert.ok(addon, 'Requires the compiled Windows native adapter');
assert.ok(expected === 'interactive' || expected === 'noninteractive', 'Specify the independently established launch context');
const native = require(addon);
assert.equal(native.isInteractiveDesktop(), expected === 'interactive');
console.log(`PASS native desktop-session observation: ${expected}`);
