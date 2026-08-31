# Agent-guided Magnitude onboarding

Use this workflow to set up Magnitude entirely through its non-interactive CLI. Keep the user in
the model-selection decision: Magnitude supplies machine-specific evidence, while the user supplies
their speed, capability, context, and storage preferences.

## 1. Establish the service

Check that the CLI is available, then install the per-user login service and start it:

```bash
magnitude --version
magnitude service install
magnitude service start
magnitude service status --json
```

`service install` enables startup for the user's operating-system session without starting the
runtime. `service start` installs or refreshes the definition as needed, starts the service, and
waits for public readiness. Both commands are safe to repeat.

Starting the service profiles the machine and assesses catalog configurations. That work may still
be reflected as initializing or assessing when the service first becomes ready.

## 2. Inspect the machine-specific catalog

Read the catalog as structured data:

```bash
magnitude catalog list --json
```

If the top-level `_tag` is `Initializing`, `localModelsReconciliationComplete` is false, or relevant
catalog rows still have `servingState._tag` equal to `Assessing`, tell the user assessment is still
running and check again. Do not use a fixed assumption about how long profiling takes, and do not
poll indefinitely without updating the user.

For model recommendations, consider catalog rows with:

- `_tag: "Catalog"`;
- `servingState._tag: "Assessed"`;
- `servingState.assessment._tag: "Fits"`; and
- present `servingState.rankingScores`.

The JSON is intentionally rich enough for a model-selection conversation. Explain the useful
evidence in ordinary language:

- `rankingScores.intelligence`, `speed`, and `fidelity` are normalized comparison scores from 0 to
  1. They are suitable for relative ranking, not absolute promises.
- `assessment.performance` contains expected generation tokens per second, lower and upper bounds,
  confidence, and the occupied context size for each estimate. These estimates describe baseline
  single-user decode and exclude prompt processing and speculative-decoding gains.
- `assessment.memory.totalRequiredBytes`, `storageBytes`, and current headroom describe memory and
  disk tradeoffs.
- `servingState.capabilities`, the assessed context profile, description, parameterization,
  quantization, license, and source URLs describe workload fit and provenance.
- `DoesNotFit`, `Incompatible`, `Failed`, or incomplete assessment states are evidence to report,
  not values to replace with guesses.

Ask what the user values before acquiring a model. At minimum establish whether they prefer faster,
balanced, or smarter output. Ask about unusually long context, vision, disk limits, or other
capability constraints only when they affect the choice. Present a short comparison of the best
fitting candidates and recommend one with a clear rationale. Preserve the exact canonical
`modelId` from the catalog.

## 3. Acquire the selected model

After the user chooses a catalog model, start or resume its acquisition:

```bash
magnitude catalog pull <model-id>
```

`pull` is Magnitude's acquisition verb: it installs a missing model, updates an outdated one, and
succeeds without duplicate work when the model is already current. There is no `catalog install`
alias.

Observe initial-install and update progress through the authoritative catalog row:

```bash
magnitude catalog list --json
```

Continue when that model's `acquisitionState._tag` is `Installed` or `UpdateAvailable`. While it is
`Installing` or `Updating`, report its stage and byte progress. If it reaches `InstallFailed` or
`UpdateFailed`, report the structured failure and stop for user direction when remediation would
expand the requested scope.

`magnitude models status --json` lists installed models and is useful after installation, but an
initial download is not yet an installed model and therefore may appear only in `catalog list`.

## 4. Load the model

Request residency and observe it separately from acquisition:

```bash
magnitude models load <model-id>
magnitude models status --json
```

The load command admits background work. Check the selected model's residency state until it is
`Ready` or `Failed`, reporting meaningful progress without busy polling. Magnitude has one active
local-model residency slot, so loading a different model may replace the current one.

## 5. Connect the current harness and install this skill

List supported harness IDs and their detected installation state:

```bash
magnitude connections list --json
```

Identify the harness in which you are currently running. Use the canonical ID from the list; if the
current harness is genuinely ambiguous, ask the user instead of configuring several harnesses.
Then connect every installed Magnitude model, select the chosen model for handoff, and refresh the
Magnitude skill in the location that harness discovers:

```bash
magnitude connections add <harness> --set-current <model-id> --install-skill
```

`--install-skill` always publishes the current bundled skill atomically. It replaces an existing
Magnitude skill at the selected physical target. Harnesses that support the shared user-wide skill
directory reuse one installation; harnesses that do not receive their supported user-wide target.
It does not copy the skill to unrelated harnesses.

Connection configures all installed Magnitude models and leaves unrelated harness settings alone.
The returned launch plan, when present, identifies the exact executable and arguments for a handoff;
do not invoke it unless the user asked you to launch or switch harnesses. A newly installed skill may
require a new harness session before that harness discovers it.

## Recovery and command discovery

Prefer `--json` for observation and decision-making. JSON success is one document on stdout; JSON
failure is one `{ "error": ... }` document on stderr with a nonzero exit status. Do not parse the
human tables when structured output is available.

Use `magnitude --help`, `magnitude <noun> --help`, or `magnitude docs cli` when the installed CLI is
newer than these instructions or a command reports an unsupported option. Do not substitute the
interactive `magnitude setup` flow while operating as the onboarding agent.
