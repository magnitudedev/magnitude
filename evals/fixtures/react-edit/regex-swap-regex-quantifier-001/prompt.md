# Fix the bug in `SimpleValues.js`

A regex quantifier was swapped, changing whitespace matching.

The issue is on line 33.

Fix the ONE regex quantifier that was swapped (between `+` and `*`). Do not modify other quantifiers.