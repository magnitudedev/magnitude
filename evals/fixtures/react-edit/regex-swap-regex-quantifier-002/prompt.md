# Fix the bug in `utils.js`

A regex quantifier was swapped, changing whitespace matching.

The issue is in the `pluralize` function.

Fix the ONE regex quantifier that was swapped (between `+` and `*`). Do not modify other quantifiers.