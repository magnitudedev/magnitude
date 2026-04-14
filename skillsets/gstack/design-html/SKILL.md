---
name: design-html
version: 1.0.0
description: |
  Design finalization: generates production-quality Pretext-native HTML/CSS.
  Works with approved mockups from /design-shotgun, CEO plans from /plan-ceo-review,
  design review context from /plan-design-review, or from scratch with a user
  description. Text actually reflows, heights are computed, layouts are dynamic.
  30KB overhead, zero deps. Smart API routing: picks the right Pretext patterns
  for each design type. Use when: "finalize this design", "turn this into HTML",
  "build me a page", "implement this design", or after any planning skill.
  Proactively suggest when user has approved a design or has a plan ready.
  Voice triggers (speech-to-text aliases): "build the design", "code the mockup", "make it real".
---

# /design-html: Pretext-Native HTML Engine

You generate production-quality HTML where text actually works correctly. Not CSS
approximations. Computed layout via Pretext. Text reflows on resize, heights adjust
to content, cards size themselves, chat bubbles shrinkwrap, editorial spreads flow
around obstacles.

---

## UX Principles: How Users Actually Behave

These principles govern how real humans interact with interfaces. They are observed
behavior, not preferences. Apply them before, during, and after every design decision.

### The Three Laws of Usability

1. **Don't make me think.** Every page should be self-evident. If a user stops
   to think "What do I click?" or "What does this mean?", the design has failed.
   Self-evident > self-explanatory > requires explanation.

2. **Clicks don't matter, thinking does.** Three mindless, unambiguous clicks
   beat one click that requires thought. Each step should feel like an obvious
   choice (animal, vegetable, or mineral), not a puzzle.

3. **Omit, then omit again.** Get rid of half the words on each page, then get
   rid of half of what's left. Happy talk (self-congratulatory text) must die.
   Instructions must die. If they need reading, the design has failed.

### How Users Actually Behave

- **Users scan, they don't read.** Design for scanning: visual hierarchy
  (prominence = importance), clearly defined areas, headings and bullet lists,
  highlighted key terms. We're designing billboards going by at 60 mph, not
  product brochures people will study.
- **Users satisfice.** They pick the first reasonable option, not the best.
  Make the right choice the most visible choice.
- **Users muddle through.** They don't figure out how things work. They wing
  it. If they accomplish their goal by accident, they won't seek the "right" way.
  Once they find something that works, no matter how badly, they stick to it.
- **Users don't read instructions.** They dive in. Guidance must be brief,
  timely, and unavoidable, or it won't be seen.

### Billboard Design for Interfaces

- **Use conventions.** Logo top-left, nav top/left, search = magnifying glass.
  Don't innovate on navigation to be clever. Innovate when you KNOW you have a
  better idea, otherwise use conventions. Even across languages and cultures,
  web conventions let people identify the logo, nav, search, and main content.
- **Visual hierarchy is everything.** Related things are visually grouped. Nested
  things are visually contained. More important = more prominent. If everything
  shouts, nothing is heard. Start with the assumption everything is visual noise,
  guilty until proven innocent.
- **Make clickable things obviously clickable.** No relying on hover states for
  discoverability, especially on mobile where hover doesn't exist. Shape, location,
  and formatting (color, underlining) must signal clickability without interaction.
- **Eliminate noise.** Three sources: too many things shouting for attention
  (shouting), things not organized logically (disorganization), and too much stuff
  (clutter). Fix noise by removal, not addition.
- **Clarity trumps consistency.** If making something significantly clearer
  requires making it slightly inconsistent, choose clarity every time.

### Navigation as Wayfinding

Users on the web have no sense of scale, direction, or location. Navigation
must always answer: What site is this? What page am I on? What are the major
sections? What are my options at this level? Where am I? How can I search?

Persistent navigation on every page. Breadcrumbs for deep hierarchies.
Current section visually indicated. The "trunk test": cover everything except
the navigation. You should still know what site this is, what page you're on,
and what the major sections are. If not, the navigation has failed.

### The Goodwill Reservoir

Users start with a reservoir of goodwill. Every friction point depletes it.

**Deplete faster:** Hiding info users want (pricing, contact, shipping). Punishing
users for not doing things your way (formatting requirements on phone numbers).
Asking for unnecessary information. Putting sizzle in their way (splash screens,
forced tours, interstitials). Unprofessional or sloppy appearance.

**Replenish:** Know what users want to do and make it obvious. Tell them what they
want to know upfront. Save them steps wherever possible. Make it easy to recover
from errors. When in doubt, apologize.

### Mobile: Same Rules, Higher Stakes

All the above applies on mobile, just more so. Real estate is scarce, but never
sacrifice usability for space savings. Affordances must be VISIBLE: no cursor
means no hover-to-discover. Touch targets must be big enough (44px minimum).
Flat design can strip away useful visual information that signals interactivity.
Prioritize ruthlessly: things needed in a hurry go close at hand, everything
else a few taps away with an obvious path to get there.

---

## Step 0: Input Detection

Ask the user how they want to start:

- **Approved mockup PNG** — they have a visual reference ready. Ask for the file path.
- **Plan document or description** — they have a CEO plan, design review notes, or written requirements. Ask them to describe what they want built.
- **Starting from scratch** — no prior context. Ask them to describe the screen they want.
- **Prior finalized HTML** — if they mention an existing HTML file from a previous session, ask: "Want to evolve it (apply changes on top, preserving edits) or start fresh?"

Route based on what the user provides. Check these cases in order:

### Case A: Approved mockup exists (design-shotgun ran)

If `APPROVED` was found, read it. Extract: approved variant PNG path, user feedback,
screen name. Also read the CEO plan if one exists (it adds strategic context).

Read `DESIGN.md` if it exists in the repo root. These tokens take priority for
system-level values (fonts, brand colors, spacing scale).

Then check for prior finalized HTML. If `FINALIZED` was also found, ask the user:
"Found a prior finalized HTML from a previous session. Want to evolve it
(apply new changes on top, preserving your custom edits) or start fresh?"

If evolve: read the existing HTML. Apply changes on top during Step 3.
If fresh or no finalized HTML: proceed to Step 1 with the approved PNG as the
visual reference.

### Case B: CEO plan and/or design variants exist, but no approved mockup

If `CEO_PLAN` or `VARIANTS` was found but no `APPROVED`:

Read whichever context exists:
- If CEO plan found: read it and summarize the product vision and design requirements.
- If variant PNGs found: show them inline using the Read tool.
- If DESIGN.md found: read it for design tokens and constraints.

Ask the user:
"Found [CEO plan | design review variants | both] but no approved design mockup.
Options:
A) Run /design-shotgun — explore design variants based on the existing plan context
B) Skip mockups — design the HTML directly from the plan context
C) I have a PNG — let me provide the path"

If A: tell the user to run /design-shotgun, then come back to /design-html.
If B: proceed to Step 1 in "plan-driven mode." Ask the user for a screen name
(e.g., "landing-page", "dashboard", "pricing") to use for the output directory.
If C: accept a PNG file path from the user and proceed with that as the reference.

### Case C: Nothing found (clean slate)

If none of the above produced any context, ask the user:
"No design context found for this project. How do you want to start?
A) Run /plan-ceo-review first — think through the product strategy before designing
B) Run /plan-design-review first — design review with visual mockups
C) Run /design-shotgun — jump straight to visual design exploration
D) Just describe it — tell me what you want and I'll design the HTML live"

If A, B, or C: tell the user to run that skill, then come back to /design-html.
If D: proceed to Step 1 in "freeform mode." Ask the user for a screen name.

### Context summary

After routing, output a brief context summary:
- **Mode:** approved-mockup | plan-driven | freeform | evolve
- **Visual reference:** path to approved PNG, or "none (plan-driven)" or "none (freeform)"
- **CEO plan:** path or "none"
- **Design tokens:** "DESIGN.md" or "none"
- **Screen name:** from approved mockup metadata, user-provided, or inferred from CEO plan

---

## Step 1: Design Analysis

> **Note:** Visual mockup generation and comparison (generating PNG mockups, running variant comparisons, vision-based spec extraction) require external design tools. If such tools are available in your environment, use them to extract a structured implementation spec from the approved PNG. If unavailable, proceed with manual visual inspection as described below.

1. **Approved mockup mode:** Read the approved PNG using the Read tool. Describe the visual layout, colors, typography, and component structure yourself. Build an implementation spec from what you observe.

2. **Plan-driven mode** (no approved PNG): Read the plan document or design review notes. Extract the described UI requirements, user flows, target audience, visual feel (dark/light, dense/spacious), content structure (hero, features, pricing, etc.), and design constraints. Build an implementation spec from the plan's prose.

3. **Freeform mode:** Ask the user what they want to build. Ask about: purpose/audience, visual feel (dark/light, playful/serious, dense/spacious), content structure, and any reference sites they like. Then build an implementation spec from their answers.

4. Read `DESIGN.md` tokens if they exist. These override any extracted values for system-level properties (brand colors, font family, spacing scale).

5. Output an "Implementation spec" summary: colors (hex), fonts (family + weights), spacing scale, component list, layout type. Generate realistic content based on the mockup or user description — never lorem ipsum.

---

## Step 2: Smart Pretext API Routing

Analyze the approved design and classify it into a Pretext tier. Each tier uses
different Pretext APIs for optimal results:

| Design type | Pretext APIs | Use case |
|-------------|-------------|----------|
| Simple layout (landing, marketing) | `prepare()` + `layout()` | Resize-aware heights |
| Card/grid (dashboard, listing) | `prepare()` + `layout()` | Self-sizing cards |
| Chat/messaging UI | `prepareWithSegments()` + `walkLineRanges()` | Tight-fit bubbles, min-width |
| Content-heavy (editorial, blog) | `prepareWithSegments()` + `layoutNextLine()` | Text around obstacles |
| Complex editorial | Full engine + `layoutWithLines()` | Manual line rendering |

State the chosen tier and why. Reference the specific Pretext APIs that will be used.

---

## Step 2.5: Framework Detection

Check if the user's project uses a frontend framework:

```bash
[ -f package.json ] && cat package.json | grep -o '"react"\|"svelte"\|"vue"\|"@angular/core"\|"solid-js"\|"preact"' | head -1 || echo "NONE"
```

If a framework is detected, ask the user: "Detected [React/Svelte/Vue] in your project. What format should the output be? Vanilla HTML (self-contained preview, recommended for first pass) or a [React/Svelte/Vue] component with Pretext hooks?"

If the user chooses framework output, ask: "TypeScript or JavaScript?"

For vanilla HTML: proceed to Step 3 with vanilla output.
For framework output: proceed to Step 3 with framework-specific patterns.
If no framework detected: default to vanilla HTML, no question needed.

---

## Step 3: Generate Pretext-Native HTML

### Pretext Source Embedding

For **vanilla HTML output**, check for a local vendored Pretext bundle:
```bash
_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
_PRETEXT_VENDOR=""
[ -n "$_ROOT" ] && [ -f "$_ROOT/vendor/pretext.js" ] && _PRETEXT_VENDOR="$_ROOT/vendor/pretext.js"
[ -n "$_PRETEXT_VENDOR" ] && echo "VENDOR: $_PRETEXT_VENDOR" || echo "VENDOR_MISSING"
```

- If `VENDOR` found: read the file and inline it in a `<script>` tag. The HTML file
  is fully self-contained with zero network dependencies.
- If `VENDOR_MISSING`: use CDN import as fallback:
  `<script type="module">import { prepare, layout, prepareWithSegments, walkLineRanges, layoutNextLine, layoutWithLines } from 'https://esm.sh/@chenglou/pretext'</script>`
  Add a comment: `<!-- FALLBACK: vendor/pretext.js missing, using CDN -->`

For **framework output**, add `@chenglou/pretext` to the project's dependencies using the appropriate package manager (bun, pnpm, yarn, or npm). Then use standard ES module imports in the component.

### HTML Generation

Write a single file using the Write tool. Save to:
`$M/reports/design-html/<screen-name>-YYYYMMDD/finalized.html`

For framework output, save to:
`$M/reports/design-html/<screen-name>-YYYYMMDD/finalized.[tsx|svelte|vue]`

**Always include in vanilla HTML:**
- Pretext source (inlined or CDN, see above)
- CSS custom properties for design tokens from DESIGN.md / Step 1 extraction
- Google Fonts via `<link>` tags + `document.fonts.ready` gate before first `prepare()`
- Semantic HTML5 (`<header>`, `<nav>`, `<main>`, `<section>`, `<footer>`)
- Responsive behavior via Pretext relayout (not just media queries)
- Breakpoint-specific adjustments at 375px, 768px, 1024px, 1440px
- ARIA attributes, heading hierarchy, focus-visible states
- `contenteditable` on text elements + MutationObserver to re-prepare + re-layout on edit
- ResizeObserver on containers to re-layout on resize
- `prefers-color-scheme` media query for dark mode
- `prefers-reduced-motion` for animation respect
- Real content extracted from the mockup (never lorem ipsum)

**Never include (AI slop blacklist):**
- Purple/blue gradients as default
- Generic 3-column feature grids
- Center-everything layouts with no visual hierarchy
- Decorative blobs, waves, or geometric patterns not in the mockup
- Stock photo placeholder divs
- "Get Started" / "Learn More" generic CTAs not from the mockup
- Rounded-corner cards with drop shadows as the default component
- Emoji as visual elements
- Generic testimonial sections
- Cookie-cutter hero sections with left-text right-image

### Pretext Wiring Patterns

Use these patterns based on the tier selected in Step 2. These are the correct
Pretext API usage patterns. Follow them exactly.

**Pattern 1: Basic height computation (Simple layout, Card/grid)**
```js
import { prepare, layout } from '@chenglou/pretext'
// Or if inlined: const { prepare, layout } = window.Pretext

// 1. PREPARE — one-time, after fonts load
await document.fonts.ready
const elements = document.querySelectorAll('[data-pretext]')
const prepared = new Map()

for (const el of elements) {
  const text = el.textContent
  const font = getComputedStyle(el).font
  prepared.set(el, prepare(text, font))
}

// 2. LAYOUT — cheap, call on every resize
function relayout() {
  for (const [el, handle] of prepared) {
    const { height } = layout(handle, el.clientWidth, parseFloat(getComputedStyle(el).lineHeight))
    el.style.height = `${height}px`
  }
}

// 3. RESIZE-AWARE
new ResizeObserver(() => relayout()).observe(document.body)
relayout()

// 4. CONTENT-EDITABLE — re-prepare when text changes
for (const el of elements) {
  if (el.contentEditable === 'true') {
    new MutationObserver(() => {
      const font = getComputedStyle(el).font
      prepared.set(el, prepare(el.textContent, font))
      relayout()
    }).observe(el, { characterData: true, subtree: true, childList: true })
  }
}
```

**Pattern 2: Shrinkwrap / tight-fit containers (Chat bubbles)**
```js
import { prepareWithSegments, walkLineRanges } from '@chenglou/pretext'

// Find the tightest width that produces the same line count
function shrinkwrap(text, font, maxWidth, lineHeight) {
  const segs = prepareWithSegments(text, font)
  let bestWidth = maxWidth
  walkLineRanges(segs, maxWidth, (lineCount, startIdx, endIdx) => {
    // walkLineRanges calls back with progressively narrower widths
    // The first call gives us the line count at maxWidth
    // We want the narrowest width that still produces this line count
  })
  // Binary search for tightest width with same line count
  const { lineCount: targetLines } = layout(prepare(text, font), maxWidth, lineHeight)
  let lo = 0, hi = maxWidth
  while (hi - lo > 1) {
    const mid = (lo + hi) / 2
    const { lineCount } = layout(prepare(text, font), mid, lineHeight)
    if (lineCount === targetLines) hi = mid
    else lo = mid
  }
  return hi
}
```

**Pattern 3: Text around obstacles (Editorial layout)**
```js
import { prepareWithSegments, layoutNextLine } from '@chenglou/pretext'

function layoutAroundObstacles(text, font, containerWidth, lineHeight, obstacles) {
  const segs = prepareWithSegments(text, font)
  let state = null
  let y = 0
  const lines = []

  while (true) {
    // Calculate available width at current y position, accounting for obstacles
    let availWidth = containerWidth
    for (const obs of obstacles) {
      if (y >= obs.top && y < obs.top + obs.height) {
        availWidth -= obs.width
      }
    }

    const result = layoutNextLine(segs, state, availWidth, lineHeight)
    if (!result) break

    lines.push({ text: result.text, width: result.width, x: 0, y })
    state = result.state
    y += lineHeight
  }

  return { lines, totalHeight: y }
}
```

**Pattern 4: Full line-by-line rendering (Complex editorial)**
```js
import { prepareWithSegments, layoutWithLines } from '@chenglou/pretext'

const segs = prepareWithSegments(text, font)
const { lines, height } = layoutWithLines(segs, containerWidth, lineHeight)

// lines = [{ text, width, x, y }, ...]
// Use for Canvas/SVG rendering or custom DOM positioning
for (const line of lines) {
  const span = document.createElement('span')
  span.textContent = line.text
  span.style.position = 'absolute'
  span.style.left = `${line.x}px`
  span.style.top = `${line.y}px`
  container.appendChild(span)
}
```

### Pretext API Reference

```
PRETEXT API CHEATSHEET:

prepare(text, font) → handle
  One-time text measurement. Call after document.fonts.ready.
  Font: CSS shorthand like '16px Inter' or 'bold 24px Georgia'.

layout(prepared, maxWidth, lineHeight) → { height, lineCount }
  Fast layout computation. Call on every resize. Sub-millisecond.

prepareWithSegments(text, font) → handle
  Like prepare() but enables line-level APIs below.

layoutWithLines(segs, maxWidth, lineHeight) → { lines: [{text, width, x, y}...], height }
  Full line-by-line breakdown. For Canvas/SVG rendering.

walkLineRanges(segs, maxWidth, onLine) → void
  Calls onLine(lineCount, startIdx, endIdx) for each possible layout.
  Find minimum width for N lines. For tight-fit containers.

layoutNextLine(segs, state, maxWidth, lineHeight) → { text, width, state } | null
  Iterator. Different maxWidth per line = text around obstacles.
  Pass null as initial state. Returns null when text is exhausted.

clearCache() → void
  Clears internal measurement caches. Use when cycling many fonts.

setLocale(locale?) → void
  Retargets word segmenter for future prepare() calls.
```

---

## Step 3.5: Live Reload Server

After writing the HTML file, start a simple HTTP server for live preview:

```bash
# Start a simple HTTP server in the output directory
_OUTPUT_DIR=$(dirname <path-to-finalized.html>)
cd "$_OUTPUT_DIR"
python3 -m http.server 0 --bind 127.0.0.1 &
_SERVER_PID=$!
_PORT=$(lsof -i -P -n | grep "$_SERVER_PID" | grep LISTEN | awk '{print $9}' | cut -d: -f2 | head -1)
echo "SERVER: http://localhost:$_PORT/finalized.html"
echo "PID: $_SERVER_PID"
```

If python3 is not available, fall back to:
```bash
open <path-to-finalized.html>
```

Tell the user: "Live preview running at http://localhost:$_PORT/finalized.html.
After each edit, just refresh the browser (Cmd+R) to see changes."

When the refinement loop ends (Step 4 exits), kill the server:
```bash
kill $_SERVER_PID 2>/dev/null || true
```

---

## Step 4: Preview + Refinement Loop

### Verification

If you have browser automation tools available, take screenshots at 375px, 768px, and 1440px viewports to verify layout. Check for:
- Text overflow (text cut off or extending beyond containers)
- Layout collapse (elements overlapping or missing)
- Responsive breakage (content not adapting to viewport)

If issues are found, fix them before presenting to the user. Otherwise, open the file in a browser and check manually at each breakpoint.

### Refinement Loop

```
LOOP:
  1. If server is running, tell user to open http://localhost:PORT/finalized.html
     Otherwise: open <path>/finalized.html

  2. If an approved mockup PNG exists, show it inline (Read tool) for visual comparison.
     If in plan-driven or freeform mode, skip this step.

  3. Ask the user (adjust wording based on mode):
     With mockup: "The HTML is live in your browser. Here's the approved mockup for comparison.
      Try: resize the window (text should reflow dynamically),
      click any text (it's editable, layout recomputes instantly).
      What needs to change? Say 'done' when satisfied."
     Without mockup: "The HTML is live in your browser. Try: resize the window
      (text should reflow dynamically), click any text (it's editable, layout
      recomputes instantly). What needs to change? Say 'done' when satisfied."

  4. If "done" / "ship it" / "looks good" / "perfect" → exit loop, go to Step 5

  5. Apply feedback using targeted Edit tool changes on the HTML file
     (do NOT regenerate the entire file — surgical edits only)

  6. Brief summary of what changed (2-3 lines max)

  7. If browser automation tools are available, re-take screenshots at 375px, 768px,
     and 1440px to confirm the fix visually before looping.

  8. Go to LOOP
```

Maximum 10 iterations. If the user hasn't said "done" after 10, ask: "We've done 10 rounds of refinement. Want to continue iterating or call it done?"

---

## Step 5: Save & Next Steps

### Design Token Extraction

If no `DESIGN.md` exists in the repo root, offer to create one from the generated HTML:

Extract from the HTML:
- CSS custom properties (colors, spacing, font sizes)
- Font families and weights used
- Color palette (primary, secondary, accent, neutral)
- Spacing scale
- Border radius values
- Shadow values

Ask the user: "No DESIGN.md found. I can extract the design tokens from the HTML we just built and create a DESIGN.md for your project, so future design runs are style-consistent automatically. Create it now, or skip?"

If they say yes: write `DESIGN.md` to the repo root with the extracted tokens.

### Save Metadata

Write `finalized.json` alongside the HTML in `$M/reports/design-html/<screen-name>-YYYYMMDD/`:
```json
{
  "source_mockup": "<approved variant PNG path or null>",
  "source_plan": "<CEO plan path or null>",
  "mode": "<approved-mockup|plan-driven|freeform|evolve>",
  "html_file": "<path to finalized.html or component file>",
  "pretext_tier": "<selected tier>",
  "framework": "<vanilla|react|svelte|vue>",
  "iterations": <number of refinement iterations>,
  "date": "<ISO 8601>",
  "screen": "<screen name>",
  "branch": "<current branch>"
}
```

### Next Steps

Ask the user: "Design finalized with Pretext-native layout. What's next? Options: copy the HTML/component into your codebase, keep refining, or done."

---

## Important Rules

- **Source of truth fidelity over code elegance.** When an approved mockup exists,
  pixel-match it. If that requires `width: 312px` instead of a CSS grid class, that's
  correct. When in plan-driven or freeform mode, the user's feedback during the
  refinement loop is the source of truth. Code cleanup happens later during
  component extraction.

- **Always use Pretext for text layout.** Even if the design looks simple, Pretext
  ensures correct height computation on resize. The overhead is 30KB. Every page benefits.

- **Surgical edits in the refinement loop.** Use the Edit tool to make targeted changes,
  not the Write tool to regenerate the entire file. The user may have made manual edits
  via contenteditable that should be preserved.

- **Real content only.** When a mockup exists, extract text from it. In plan-driven mode,
  use content from the plan. In freeform mode, generate realistic content based on the
  user's description. Never use "Lorem ipsum", "Your text here", or placeholder content.

- **One page per invocation.** For multi-page designs, run /design-html once per page.
  Each run produces one HTML file.
