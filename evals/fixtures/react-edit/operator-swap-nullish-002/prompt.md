# Fix the bug in `ReactFlightServerConfigTurbopackBundler.js`

A nullish coalescing operator was swapped.

The issue is in the `resolveClientReferenceMetadata` function.

Use the intended nullish/logical operator.