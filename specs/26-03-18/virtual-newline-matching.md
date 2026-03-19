# Virtual Newline Matching for File Edits

## Problem

Content inside XML tool call tags is interpreted literally — no trimming or stripping. This gives models full control over newlines and whitespace.

However, models naturally format XML with newlines after opening tags and before closing tags:

```xml
<old>
content here
</old>
```

This produces `oldString = "\ncontent here\n"`. When the content is in the **middle** of a file, these newlines match the real newlines surrounding the content. But when the content is at the **start or end of the file**, there may not be a real newline to match — causing the edit to fail even though the model correctly identified the content.

## Solution: Virtual File Boundaries

When an exact match fails, we retry matching against a virtual version of the file with `\n` prepended and appended:

```
virtualFile = '\n' + realFile + '\n'
```

These virtual newlines exist **only for matching purposes** — they are never written to the file. They absorb the model's formatting newlines at file boundaries, making edits work regardless of whether the content is in the middle, at the start, or at the end of the file.

## Algorithm

```
1. Try matching oldString against the real file content
2. If found → replace normally (no virtual matching needed)
3. If not found → try matching against virtualFile ('\n' + content + '\n')
4. If found in virtualFile:
   a. Determine which virtual boundaries were consumed by the match
   b. Map the match position back to the real file
   c. Clip the corresponding formatting newlines from newString
   d. Apply the replacement to the real file
5. If not found in virtualFile either → error "not found"
```

### Virtual matching is a fallback

Real matches always take priority. Virtual matching is only attempted when no real match exists. This prevents false positives where a virtual boundary creates an extra match the model didn't intend.

### Clipping newString

When a virtual boundary is consumed during matching, the corresponding newline in `newString` is treated as a formatting newline and clipped:

| Match consumed... | newString clipping |
|---|---|
| Virtual leading `\n` | Remove leading `\n` from newString (if present) |
| Virtual trailing `\n` | Remove trailing `\n` from newString (if present) |
| Both | Remove both (if present) |
| Neither | No clipping |

### Core assumption

**The model is consistent about formatting newlines between `<old>` and `<new>` within a single edit call.** If `<old>` has a formatting newline after the opening tag, `<new>` does too. The clipping applied to `newString` is driven by what happened during the `oldString` match — if old's leading `\n` was absorbed by a virtual boundary, new's leading `\n` is also clipped.

Intentional content differences (e.g. adding a trailing newline) are expressed through the content itself, not through inconsistent formatting.

## Examples

### Middle of file (no virtual matching needed)
```
File:    "aaa\ncontent\nzzz"
Old:     "\ncontent\n"
New:     "\nreplaced\n"
Match:   real match at position 3
Result:  "aaa\nreplaced\nzzz"
```

### Content at start of file
```
File:    "content\nzzz"
Old:     "\ncontent\n"       (model's formatting newlines)
New:     "\nreplaced\n"
Real:    no match (file doesn't start with \n)
Virtual: "\ncontent\nzzz\n" → match at position 0
         Leading \n consumed by virtual boundary
Clip:    "\nreplaced\n" → "replaced\n"
Result:  "replaced\nzzz"
```

### Content at end of file
```
File:    "aaa\ncontent"
Old:     "\ncontent\n"
New:     "\nreplaced\n"
Real:    no match (file doesn't end with \n)
Virtual: "\naaa\ncontent\n" → match found
         Trailing \n consumed by virtual boundary
Clip:    "\nreplaced\n" → "\nreplaced"
Result:  "aaa\nreplaced"
```

### Entire file replacement
```
File:    "content"
Old:     "\ncontent\n"
New:     "\nnew content\n"
Real:    no match
Virtual: "\ncontent\n" → match spans entire virtual file
         Both boundaries consumed
Clip:    "\nnew content\n" → "new content"
Result:  "new content"
```

### Adding a trailing newline at EOF
```
File:    "hello\nworld"
Old:     "world"            (inline, no formatting newlines)
New:     "world\n"
Match:   real match — no virtual needed
Result:  "hello\nworld\n"   (trailing newline added)
```

### Inline match (no formatting newlines)
```
File:    "hello\nworld"
Old:     "world"
New:     "earth"
Match:   real match
Result:  "hello\nearth"
```

## Edge Cases

### Zero-length real match region
If the entire `oldString` is consumed by virtual boundaries (e.g. `oldString = "\n"`), the virtual match is rejected. The real match region must be non-empty.

### Empty oldString
Rejected immediately with an error — prevents infinite loops in substring search.

### replaceAll with virtual matching
- Real matches and virtual matches are never mixed
- If real matches exist, only real matches are used
- Virtual matches are only attempted when zero real matches exist
- Each virtual match gets independent clipping based on which boundary it consumed
- Replacements are applied in reverse order (end→start) to preserve positions

## Why This Design

1. **No trimming** — content is literal, giving models full control over newlines
2. **Tolerant at boundaries** — the model's natural XML formatting doesn't break edits at file start/end
3. **Exact in the middle** — interior matches are always exact, no fuzzy behavior
4. **Fallback only** — virtual matching never interferes with real matches
5. **Symmetric clipping** — assumes consistent formatting between `<old>` and `<new>`, clips both the same way

## Implementation

- `packages/agent/src/util/edit.ts` — `validateAndApply()` with `tryVirtualMatch()` fallback
- `packages/agent/src/util/__tests__/edit-virtual-newlines.test.ts` — 40 tests covering all scenarios