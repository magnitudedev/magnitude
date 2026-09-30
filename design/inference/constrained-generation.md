---
applies_to:
  - inference/engine/grammar/**
  - inference/engine/src/worker/binding.rs
  - inference/engine/chat/src/preparation.rs
---

# Constrained generation

## Ownership

One layer owns constrained decoding from grammar source to per-token masks.
Its input is GBNF, from a chat template or a caller, and its output is a
generation constraint.

- Compilation is language work and needs no tokenizer. The host compiles
  during request preparation, so an inadmissible caller grammar is an invalid
  request before admission. Compiled grammars are cached per model by source.
- Binding is tokenizer work. The worker binds a compiled grammar to its own
  vocabulary and the grammar prefix the prompt already contains, and caches
  bound grammars.
- Nothing outside the layer depends on how grammars are executed: the matcher
  library, its grammar dialect and lexeme structure stay inside it.

GBNF remains the external format. The constrained language itself (argument
order, raw-string policy, reasoning format) is decided by the templates, never
by this layer.

## Guarantees

- **Exact.** The compiled grammar accepts exactly the GBNF language, and a
  mask admits a token exactly when some completion of the language remains.
  Rules that can match nothing are removed before compilation.
- **Bounded.** Compiled size is at most a fixed multiple of the source size.
  Shared structure is referenced, never copied, except where flattening
  copies structure under an explicit allowance.
- **Local degradation.** A part of a grammar that greedy lexing cannot scan
  exactly is scanned one character at a time. That applies to that part only,
  never to the whole grammar.
- **Template grammars are fully lexical.** Grammars produced by the templates
  compile with no character-level parts.
- **Observable.** Every compilation reports its size, lexical shape and every
  degradation; the host records the report.

## Compilation model

The matcher parses with Earley over lexemes and scans each lexeme with a DFA,
greedily and without backtracking: a lexeme ends only when the next byte can
continue no allowed lexeme, and every lexeme matching that text is emitted.
Two lexemes matching the same text is therefore safe; a string is lost only
when a completed lexeme is abandoned because another allowed lexeme continues
over a byte that may follow it.

A compiled grammar is a lexical plan:

- **Terminals.** A rule whose language is regular (non-recursive over regular
  rules, or recursive only in tail position, as delimiter scanners are) is a
  named regular language defined by reference to other terminals.
- **Earley rules.** Recursive rules and the root keep grammar rules.
  Non-recursive rules that contain recursion are flattened into them so that
  a scanner and the delimiter that ends it stay in one lexeme. Flattening is
  bounded: it may at most double the call sites, and no boundary may start
  more than a fixed number of lexemes; beyond that, the largest copied rules
  stay calls.
- **Lexemes.** A lexeme is every regular path between two parse boundaries.
  Boundaries are rule entries and exits, private endpoints of each call (so a
  path that bypasses a call never ends where the call starts), and heads of
  repetitions whose body contains a call (so a lexeme spans one iteration).
  Text before such a repetition that continues over the text its body leads
  with, yet never contains it, is a scanner before its delimiter: the
  repetition's first iteration is built before its head, so both stay in one
  lexeme.

Each multi-character lexeme is certified exact against every lexeme that can
be allowed with it. Allowed sets are tracked per calling context; call sites
predicted at the same input position share their callee's continuations, as
Earley shares the callee. A lexeme that cannot be certified is scanned one
character at a time, and certification repeats until none remains. Questions
the certifier cannot answer within its work budget count as conflicts.

## Execution

Masks are computed per token from the bound matcher. Forks and staged
successors copy parser state and share the request's lexer cache; bound
grammars cached across requests give each request its own lexer. The
matcher's default work limits apply; exceeding one indicates a defect, not a
budget to raise.

## Acceptance

- The compiled grammar and an all-character rendering of the same plan agree
  on generated strings, near misses and masks.
- Template grammars across tool choices, reasoning modes, response formats and
  argument schemas, including any-order argument lattices and recursive
  argument values, compile with no character-level parts.
- Per-token mask cost does not grow with the number of offered tools.
