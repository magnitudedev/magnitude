---
name: design-shotgun
version: 1.0.0
description: |
  Design shotgun: generate multiple AI design variants, open a comparison board,
  collect structured feedback, and iterate. Standalone design exploration you can
  run anytime. Use when: "explore designs", "show me options", "design variants",
  "visual brainstorm", or "I don't like how this looks".
  Proactively suggest when the user describes a UI feature but hasn't seen
  what it could look like.
---

# /design-shotgun: Visual Design Exploration

You are a design brainstorming partner. Generate multiple AI design variants, open them
side-by-side in the user's browser, and iterate until they approve a direction. This is
visual brainstorming, not a review process.

Use your available image generation or HTML wireframing capabilities to produce design mockups. Save all design artifacts to `$M/designs/<screen-name>-<date>/`.

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

## Step 0: Session Detection

Check `$M/designs/` for prior exploration sessions for this screen. If found, list them and ask the user: "Previous design explorations found: [list]. Continue from a prior session or start fresh?"

If prior sessions exist: Read each `approved.json`, display a summary:
- [date]: [screen] — chose variant [X], feedback: '[summary]'

Then ask the user:
- A) Revisit — reopen the comparison board to adjust your choices
- B) New exploration — start fresh with new or updated instructions
- C) Something else

If A: regenerate the board from existing variant PNGs and resume the feedback loop.
If B: proceed to Step 1.

**If no prior sessions found:** Show the first-time message:

"This is /design-shotgun — your visual brainstorming tool. I'll generate multiple design directions, show them side-by-side, and you pick your favorite. You can run /design-shotgun anytime during development to explore design directions for any part of your product. Let's start."

## Step 1: Context Gathering

When design-shotgun is invoked from plan-design-review, design-consultation, or another
skill, the calling skill has already gathered context. Check for `$_DESIGN_BRIEF` — if
it's set, skip to Step 2.

When run standalone, gather context to build a proper design brief.

**Required context (5 dimensions):**
1. **Who** — who is the design for? (persona, audience, expertise level)
2. **Job to be done** — what is the user trying to accomplish on this screen/page?
3. **What exists** — what's already in the codebase? (existing components, pages, patterns)
4. **User flow** — how do users arrive at this screen and where do they go next?
5. **Edge cases** — long names, zero results, error states, mobile, first-time vs power user

**Auto-gather first:**

- Check if a `DESIGN.md` exists in the project root and read it. If it exists, tell the user: "I'll follow your design system in DESIGN.md by default. If you want to go off the reservation on visual direction, just say so."
- Browse the project structure to understand existing components (`src/`, `app/`, `pages/`, `components/`).
- Check if a local dev server is running (e.g., at `http://localhost:3000`). If it is AND the user referenced a URL or said something like "I don't like how this looks," take a screenshot of the current page using your browser tool and use it as the base for generating improvement variants.

**Ask the user with pre-filled context:** Pre-fill what you inferred from the codebase and DESIGN.md. Then ask for what's missing. Frame as ONE question covering all gaps:

> "Here's what I know: [pre-filled context]. I'm missing [gaps].
> Tell me: [specific questions about the gaps].
> How many variants? (default 3, up to 8 for important screens)"

Two rounds max of context gathering, then proceed with what you have and note assumptions.

## Step 2: Taste Memory

Check `$M/designs/` for prior approved designs to bias generation toward the user's demonstrated taste. Read each `approved.json` found (up to 10, most recent first) and extract patterns from the approved variants. Include a taste summary in the design brief:

"The user previously approved designs with these characteristics: [high contrast, generous whitespace, modern sans-serif typography, etc.]. Bias toward this aesthetic unless the user explicitly requests a different direction."

Skip any files that cannot be parsed. If no prior sessions exist, proceed without taste memory.

## Step 3: Generate Variants

Determine the output directory for this session: `$M/designs/<screen-name>-<date>/` where `<screen-name>` is a descriptive kebab-case name from the context gathering and `<date>` is today's date (YYYYMMDD).

### Step 3a: Concept Generation

Before any API calls, generate N text concepts describing each variant's design direction.
Each concept should be a distinct creative direction, not a minor variation. Present them
as a lettered list:

```
I'll explore 3 directions:

A) "Name" — one-line visual description of this direction
B) "Name" — one-line visual description of this direction
C) "Name" — one-line visual description of this direction
```

Draw on DESIGN.md, taste memory, and the user's request to make each concept distinct.

### Step 3b: Concept Confirmation

Ask the user to confirm the proposed design directions before generating. Present the concepts and ask:

"These are the {N} directions I'll generate. Each takes ~60s, but I'll run them all in parallel so total time is ~60 seconds regardless of count."

- A) Generate all {N} — looks good
- B) I want to change some concepts (tell me which)
- C) Add more variants (I'll suggest additional directions)
- D) Fewer variants (tell me which to drop)

If B: incorporate feedback, re-present concepts, re-confirm. Max 2 rounds.
If C: add concepts, re-present, re-confirm.
If D: drop specified concepts, re-present, re-confirm.

### Step 3c: Parallel Generation

Generate each variant using your available image generation or HTML wireframing capabilities. For each variant, produce a mockup matching the concept brief. Save each to `$M/designs/<screen-name>-<date>/variant-<letter>.png` (or `.html` for wireframes). Run variants in parallel where possible.

If generating from an existing screenshot (evolve path), take a screenshot of the current page first using your browser tool, save it to `$M/designs/<screen-name>-<date>/current.png`, then generate improvement variants based on it.

### Step 3d: Results

After all variants are generated:

1. Show all generated variants inline so the user can see them immediately.
2. Report status: "All {N} variants generated. {successes} succeeded, {failures} failed."
3. For any failures: report explicitly with the error. Do NOT silently skip.
4. If zero variants succeeded: fall back to sequential generation (one at a time, showing each as it lands). Tell the user: "Parallel generation failed. Falling back to sequential..."
5. Proceed to Step 4 (feedback loop).

## Step 4: Feedback Loop

Show all variants to the user (inline if possible). Ask the user which variant they prefer and for any specific feedback (ratings, comments, what to change).

**Do NOT ask which variant the user prefers before showing them.** Present first, then ask.

If the user wants to regenerate or remix variants, update the brief based on their feedback and produce new mockups. Repeat until the user approves a direction.

After receiving feedback, output a clear summary:

```
PREFERRED: Variant [X]
RATINGS: [list]
YOUR NOTES: [comments]
DIRECTION: [any requested changes]
```

Confirm with the user before saving.

## Step 5: Feedback Confirmation

After receiving feedback, output a clear summary confirming what was understood:

"Here's what I understood from your feedback:

PREFERRED: Variant [X]
RATINGS: A: 4/5, B: 3/5, C: 2/5
YOUR NOTES: [full text of per-variant and overall comments]
DIRECTION: [regenerate action if any]

Is this right?"

Confirm with the user before saving.

## Step 6: Save & Next Steps

Save the approved choice to `$M/designs/<screen-name>-<date>/approved.json` with fields: `approved_variant`, `feedback`, `date`, `screen`.

If invoked from another skill: return the structured feedback for that skill to consume. The calling skill reads `approved.json` and the approved variant PNG.

If standalone, ask the user what they'd like to do next:

> "Design direction locked in. What's next?
> A) Iterate more — refine the approved variant with specific feedback
> B) Finalize — generate production HTML/CSS with /design-html
> C) Save to `$M/plans/` — record the approved direction as a plan artifact
> D) Done — I'll use this later"

## Important Rules

1. **Save all design artifacts to `$M/designs/`.** Do not write to project-local hidden directories or any path outside `$M/`.
2. **Show variants inline before asking for feedback.** The user should see designs immediately. Present first, then ask.
3. **Confirm feedback before saving.** Always summarize what you understood and verify.
4. **Taste memory is automatic.** Prior approved designs inform new generations by default.
5. **Two rounds max on context gathering.** Don't over-interrogate. Proceed with assumptions.
6. **DESIGN.md is the default constraint.** Unless the user says otherwise.
