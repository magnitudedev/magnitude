# Fix the bug in `registerDevToolsEventLogger.js`

Optional chaining was removed from a property access.

The issue is on line 34.

Restore the optional chaining operator (`?.`) at the ONE location where it was removed. Do not add optional chaining elsewhere.