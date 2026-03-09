# Fix the bug in `ReactDOMSelection.js`

A boolean literal is inverted.

The issue is in the `getModernOffsetsFromPoints` function.

Flip the boolean literal to the intended value.