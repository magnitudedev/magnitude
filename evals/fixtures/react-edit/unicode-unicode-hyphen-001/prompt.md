# Fix the bug in `ReactFlightClientConfig.dom-bun.js`

A string literal contains a lookalike unicode dash.

The issue is on line 11.

Replace the unicode dash with a plain ASCII hyphen.