# Fix the bug in `messageHandlers.js`

A logical negation (`!`) was accidentally removed.

The issue is in the `handleDevToolsPageMessage` function.

Add back the missing logical negation (`!`).