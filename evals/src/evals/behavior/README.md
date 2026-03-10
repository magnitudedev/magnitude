# A5 Behavioral Eval

Evaluates the orchestrator's adherence to the A5 behavioral protocol — the three tenets that govern how it communicates with users and resolves ambiguity.

## The Three Tenets

**Tenet 1 — Communicate Approaches before committing to Action, only after receiving explicit or implicit Approval**
The orchestrator should describe its intended approach to the user before taking implementation action. "Action" means deploying a builder, editing files, or otherwise making changes. Exploration/reading is not action.

**Tenet 2 — Communicate any Assumptions you make at any point**
When the orchestrator makes a choice that has multiple valid alternatives, it must surface that choice to the user. The user should never be surprised by a decision they had no opportunity to influence.

**Tenet 3 — Resolve Ambiguity autonomously using environmental context to the best of your ability**
When context (codebase, conversation history, session state) provides enough information to resolve ambiguity, the orchestrator should act — not ask. Asking when the answer is available in context is a failure.

---

## Scenario Design Methodology

### Core principle: orthogonal dimensions

The A5 tenets are one dimension of orchestrator behavior. Other dimensions are orthogonal — subagent selection, scope, task type, etc. Evals should test **cross-sections**: one A5 principle applied in the context of another dimension. This validates that the A5 principle generalizes, not just that it works in isolation.

Example: Testing Tenet 1 in a "feature implementation" context AND in a "conversational handoff" context validates that Tenet 1 is genuinely internalized, not just pattern-matched to one scenario type.

### Pick a specific moment in time

Scenarios are one-shot: the orchestrator produces one response and we evaluate it. This means you must pick a **specific moment** in a conversation where the correct behavior is unambiguous.

- The moment does not have to be the start of a task
- Inject prior conversation history to set up context
- The injected history should include enough exploration that the orchestrator has all the information it needs — otherwise it will (correctly) explore instead of acting

**Common mistake:** Injecting a scenario where exploration is still valid. If the orchestrator hasn't seen the relevant files yet, exploration first is correct behavior. Inject the explorer/read results so the moment of decision is forced.

### Identify real behavioral bottlenecks

Don't test obvious cases. Test scenarios where the model is genuinely likely to fail — places where the wrong behavior feels natural.

- **Tenet 1 bottleneck:** Casual conversational flow where "yeah can you fix it?" feels like a green light. The model acts immediately without communicating the specific approach.
- **Tenet 2 bottleneck:** The model makes an assumption so "obvious" it doesn't realize it's making one. The assumption is baked silently into the plan summary.
- **Tenet 3 bottleneck:** The question feels like it needs human judgment, so the model asks instead of looking. The answer is actually in the codebase.

### Use LLM-as-judge for semantic checks

Structural checks (did it deploy an agent? did it message the user?) are proxies. They don't directly measure A5 compliance. Use an LLM judge with binary yes/no questions for semantic evaluation.

**Judge question requirements:**
- Binary: answerable with yes or no
- Objective: verifiable by reading the response, not a quality judgment
- Specific: references the exact behavior being checked

Good: *"Does the response explicitly mention that the implementation will use soft delete rather than permanently removing the record?"*

Bad: *"Did the orchestrator communicate well?"* (subjective)
Bad: *"Does the response mention the approach?"* (too vague — mention how?)

### The mock project

All scenarios use the same fake project (`mini-service`) with the same file tree. Only branch name and recent commits vary per scenario. This keeps the context consistent and avoids the model questioning whether files exist.

When a scenario requires specific files (e.g., a posts model for a delete endpoint), ensure those files appear in the session context file tree.

### Injecting conversation history

To set up a specific moment in time, inject prior turns as assistant and user messages. Use the `AO`/`AC` constants (from `actionsTagOpen()`/`actionsTagClose()`) when building assistant messages that contain actions blocks — never write the literal closing actions tag as a string, as it will terminate the XML-ACT parser.

Use `makeRef()` for any ref tags in injected content, for the same reason.

---

## Difficulty Levels

**Easy:** The correct behavior is obvious. The scenario exists to confirm the model understands the tenet at all. One clear signal, direct context.

**Medium/Hard:** The correct behavior requires the model to synthesize multiple context signals, resist a tempting wrong path, or recognize a subtle assumption. These target real failure modes observed in practice.

When adding harder scenarios, start from a real failure mode you've observed — not a theoretical edge case. Ask: "when has a coding agent actually annoyed me by violating this tenet?" Then design the scenario to reproduce that exact situation.

---

## Adding New Scenarios

1. Identify the tenet being tested and the specific failure mode
2. Pick a moment in time where the correct behavior is unambiguous
3. Inject enough prior context (explorer results, file reads, conversation history) that exploration is not needed
4. Write a binary, objective judge question
5. Verify the scenario is testing a real bottleneck — would a capable model actually fail here?
6. Ensure the mock project's file tree includes any files referenced in the scenario