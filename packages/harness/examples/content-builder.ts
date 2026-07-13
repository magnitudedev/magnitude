/**
 * Example: ContentBuilder for assembling mixed text+image content
 *
 * ContentBuilder produces ToolResultPart[] with automatic text coalescing —
 * consecutive text pushes merge into a single TextPart, while images are
 * kept as separate ImageParts.
 */

import { ContentBuilder } from '../src'

// ── Basic text coalescing ────────────────────────────────────────────

const builder = new ContentBuilder()
builder.pushText('<name>index.ts</name>\n')
builder.pushText('<size>1024</size>')

const parts = builder.build()
// → [{ _tag: 'TextPart', text: '<name>index.ts</name>\n<size>1024</size>' }]
// Two pushText calls coalesced into one TextPart

// ── Mixed text and images ────────────────────────────────────────────

const mixed = new ContentBuilder()
mixed.pushText('<screenshot>\n')
mixed.pushPart({
  _tag: 'ImagePart',
  data: 'iVBORw0KGgo...', // base64 image data
  mediaType: 'image/png',
})
mixed.pushText('\n</screenshot>\n')
mixed.pushText('<title>Homepage</title>')

const mixedParts = mixed.build()
// → [
//   { _tag: 'TextPart', text: '<screenshot>\n' },
//   { _tag: 'ImagePart', data: 'iVBORw0KGgo...', mediaType: 'image/png' },
//   { _tag: 'TextPart', text: '\n</screenshot>\n<title>Homepage</title>' },
// ]
// Text before and after the image are separate TextParts.
// The two text pushes after the image are coalesced.

// ── pushParts for bulk insertion ─────────────────────────────────────

const bulk = new ContentBuilder()
bulk.pushParts([
  { _tag: 'TextPart', text: 'first ' },
  { _tag: 'TextPart', text: 'second' },
  { _tag: 'ImagePart', data: 'abc123', mediaType: 'image/jpeg' },
  { _tag: 'TextPart', text: 'after image' },
])

const bulkParts = bulk.build()
// → [
//   { _tag: 'TextPart', text: 'first second' },     ← coalesced
//   { _tag: 'ImagePart', data: 'abc123', mediaType: 'image/jpeg' },
//   { _tag: 'TextPart', text: 'after image' },
// ]

// ── Checking for content ─────────────────────────────────────────────

const empty = new ContentBuilder()
console.log(empty.hasContent()) // false

empty.pushText('hello')
console.log(empty.hasContent()) // true

// Empty strings are ignored
const stillEmpty = new ContentBuilder()
stillEmpty.pushText('')
console.log(stillEmpty.hasContent()) // false
