# Fix the bug in `backendManager.js`

A string literal contains a lookalike unicode dash.

The issue is in the `updateRequiredBackends` function.

Replace the unicode dash with a plain ASCII hyphen.