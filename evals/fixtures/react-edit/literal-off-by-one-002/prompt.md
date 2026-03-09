# Fix the bug in `ReactFlightDOMClientNode.js`

A numeric boundary has an off-by-one error.

The issue is in the `createFromNodeStream` function.

Fix the off-by-one error in the numeric literal or comparison.