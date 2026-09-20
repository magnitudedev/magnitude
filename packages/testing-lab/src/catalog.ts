import { planExecution } from "./execution-plan"
import { Effect, Option, Schema } from "effect"
import { CaseId, Harness, InvalidInput, PlannedCase, RunPlan, RunRequest, Selection, Suite, Target, TargetId } from "./domain"

const target = (os: Target["os"], version: string, arch: Target["arch"], backend: Target["backend"], hardware: Target["hardware"]): Target =>
  Schema.decodeUnknownSync(Target)({ id: `${os}-${version}-${arch}-${backend}-${hardware}`,
    os, version, arch, backend, hardware,
    provider: os === "macos" ? "namespace" : os === "dgx-os" ? "spark" : "azure",
    artifactHost: os === "macos" ? "darwin-arm64" : os === "windows" ? "windows-x64-msvc" : `linux-${arch}-gnu`,
    packageFormat: os === "macos" ? "dmg" : os === "windows" ? "exe" : os === "fedora" || os === "redhat" ? "rpm" : "deb",
  })

export const targets: readonly Target[] = [
  ...["15", "26"].flatMap(v => [target("macos", v, "arm64", "cpu", "apple-silicon"), target("macos", v, "arm64", "metal", "apple-silicon")]),
  ...["10", "11"].flatMap(v => [
    target("windows", v, "x64", "cpu", "intel"), target("windows", v, "x64", "cpu", "amd"),
    target("windows", v, "x64", "cpu", "a10"), target("windows", v, "x64", "cuda", "a10"),
  ]),
  target("windows", "11", "x64", "cpu", "rtx-pro-6000"), target("windows", "11", "x64", "cuda", "rtx-pro-6000"),
  ...([["ubuntu", "24.04"], ["debian", "13"], ["fedora", "44"], ["redhat", "10"]] as const).flatMap(([os, v]) => [
    target(os, v, "x64", "cpu", "intel"), target(os, v, "x64", "cpu", "amd"), target(os, v, "arm64", "cpu", "arm"),
    ...(["a10", "rtx-pro-6000"] as const).flatMap(hw => [target(os, v, "x64", "cpu", hw), target(os, v, "x64", "cuda", hw)]),
  ]),
  target("dgx-os", "7", "arm64", "cpu", "dgx-spark"), target("dgx-os", "7", "arm64", "cuda", "dgx-spark"),
]

export const prTargetIds = [
  "macos-15-arm64-cpu-apple-silicon", "macos-26-arm64-metal-apple-silicon",
  "windows-10-x64-cpu-intel", "windows-11-x64-cpu-amd", "windows-10-x64-cuda-a10",
  "windows-11-x64-cuda-a10", "windows-11-x64-cuda-rtx-pro-6000",
  "ubuntu-24.04-x64-cpu-intel", "ubuntu-24.04-arm64-cpu-arm", "ubuntu-24.04-x64-cuda-a10",
  "ubuntu-24.04-x64-cuda-rtx-pro-6000", "dgx-os-7-arm64-cuda-dgx-spark",
].map(id => TargetId.make(id))

/** Case identity is stable; titles describe an assertion rather than an implementation hook. */
const suiteCases: Readonly<Record<Suite, readonly string[]>> = {
  package: ["Compile declared inputs", "Produce final packages", "Versions and architectures agree", "Owned runtime dependencies are complete", "Validate package signatures and trust"],
  install: ["Download exact bytes and verify digest", "Install through the supported package manager", "Launch the installed application", "Verify CLI path and service ownership", "Reject a corrupt candidate before installation"],
  app: ["Complete setup to a ready application", "Search catalog and open model details", "Download the pinned model through the application", "Persist settings across relaunch", "Manage harness Connections without clobbering unrelated config", "Hide, reopen and quit the application", "Surface actionable harness and service errors"],
  endpoint: ["Discover the installed model", "Generate a nonstreamed answer", "Generate a complete valid stream", "Complete a tool call and result follow-up", "Reject invalid model and malformed requests without crashing", "Attest the requested backend and device"],
  harness: ["Consume the product-generated harness connection", "Select the served model", "Generate a streamed answer", "Continue the same session", "Read and edit a bounded fixture through actual tools", "Resume a persisted transcript", "Drive the TUI and interrupt generation"],
  recovery: ["Cancel generation and generate again", "Recover an interrupted model download", "Generate offline from cached model bytes", "Recover an owned worker fault", "Stop and reload the model", "Restart without duplicate owning services"],
  cli: ["Run the exact bundled CLI version", "Inspect help, hardware and service status", "Use catalog and model lifecycle commands", "Manage harness connections through the bundled CLI", "Verify exit behavior and interruption", "Run without ambient development runtimes"],
  update: ["Install the previous version with persistent state", "Update using the normal application updater", "Verify replacement versions and bytes", "Preserve settings and generate after update", "Reject a corrupt or untrusted update", "Recover an interrupted update"],
  uninstall: ["Remove package-owned registration and launchers", "Leave no owned processes or runnable login entry", "Preserve or delete user data according to product policy", "Reinstall without stale service state"],
}
const prefixes: Readonly<Record<Suite, string>> = { package: "P", install: "I", app: "A", endpoint: "E", harness: "H", recovery: "R", cli: "C", update: "U", uninstall: "X" }
const prerequisites: Readonly<Record<string, readonly string[]>> = {
  P2: ["P1"], P3: ["P2", "I3"], P4: ["P2"], P5: ["P2"],
  I1: ["P2"], I2: ["I1"], I3: ["I2"], I4: ["I3"], I5: ["I1"],
  A1: ["I3"], A2: ["A1"], A3: ["A2"], A4: ["A1"], A5: ["A3"], A6: ["A1"], A7: ["A3"],
  E1: ["A3"], E2: ["E1", "E6"], E3: ["E1", "E6"], E4: ["E1", "E6"], E5: ["E1"], E6: ["E1"],
  H1: ["A5", "E6"], H2: ["H1"], H3: ["H2"], H4: ["H3"], H5: ["H2"], H6: ["H4"], H7: ["H2"],
  R1: ["E3"], R2: ["A3"], R3: ["E2"], R4: ["E2"], R5: ["E2"], R6: ["I3"],
  C1: ["I2"], C2: ["C1", "A1"], C3: ["C1", "A3"], C4: ["C1", "A5"], C5: ["C1", "A1"], C6: ["C1"],
  U1: ["P2"], U2: ["U1"], U3: ["U2"], U4: ["U3"], U5: ["U1"], U6: ["U1"],
  X1: ["I2"], X2: ["X1"], X3: ["X1"], X4: ["X2", "X3"],
}
export const cases: readonly PlannedCase[] = Object.entries(suiteCases).flatMap(([suite, titles]) => titles.map((title, index) => ({
  id: CaseId.make(`${prefixes[suite as Suite]}${index + 1}`), suite: suite as Suite, title,
  timeoutSeconds: suite === "package" ? 3600 : suite === "recovery" && index === 1 ? 2400 : suite === "update" || suite === "harness" && index === 6 || title.includes("model") ? 900 : 300,
  harness: Option.none<Harness>(), prerequisites: (prerequisites[`${prefixes[suite as Suite]}${index + 1}`] ?? []).map(id => CaseId.make(id)),
})))
const quickIds = new Set(["P1", "P2", "P3", "P4", "P5", "I1", "I2", "I3", "I4", "A1", "A2", "A3", "A5", "E1", "E3", "E6", "H1", "H2", "H3", "H5", "C1", "C2"])
const updateRepresentatives = new Set(["macos-26-arm64-metal-apple-silicon", "windows-11-x64-cuda-a10", "ubuntu-24.04-x64-cuda-a10"])

export const findTarget = (id: TargetId) => Effect.fromNullable(targets.find(t => t.id === id)).pipe(
  Effect.mapError(() => new InvalidInput({ message: `Unknown target: ${id}` })),
)
export const selectedHarnesses = (selection: typeof Selection.Type): readonly Harness[] => selection.kind === "custom" ? selection.harnesses
  : Option.getOrElse(selection.harnesses, (): readonly Harness[] => selection.profile === "quick" ? ["pi"] : ["pi", "opencode", "hermes"])

export const planRun = (request: RunRequest) => Effect.gen(function* () {
  if (request.mode !== "verify") return yield* new InvalidInput({ message: "Warm worker reuse is not implemented; use verify mode" })
  if (Option.isSome(request.updateFrom)) return yield* new InvalidInput({ message: "Historical release migration is not implemented; omit updateFrom to test a source-built update pair" })
  const selection = request.selection
  if (selection.kind === "profile" && Option.isSome(selection.harnesses) && selection.profile !== "quick") {
    return yield* new InvalidInput({ message: "Harness overrides apply only to quick; use custom selection to narrow other profiles" })
  }
  const harnesses = selectedHarnesses(selection)
  if (new Set(harnesses).size !== harnesses.length) return yield* new InvalidInput({ message: "Duplicate harnesses in selection" })
  if (selection.kind === "profile" && Option.isSome(selection.target) && selection.profile !== "quick") {
    return yield* new InvalidInput({ message: "Only quick accepts one target override; use a custom selection to narrow other profiles" })
  }
  if (selection.kind === "profile" && selection.profile === "quick" && Option.isNone(selection.target)) {
    return yield* new InvalidInput({ message: "Quick requires an explicit target (the CLI can resolve the local target)" })
  }
  if (selection.kind === "profile" && selection.profile === "release" && (request.input.kind !== "artifacts" || request.mode !== "verify" || request.trust === "untrusted-ci")) {
    return yield* new InvalidInput({ message: "Release requires final artifacts, verify isolation and trusted execution" })
  }
  const ids = selection.kind === "custom" ? selection.targets : selection.profile === "quick" ? [Option.getOrThrow(selection.target)]
    : selection.profile === "pr" ? prTargetIds : targets.map(t => t.id)
  if (new Set(ids).size !== ids.length) return yield* new InvalidInput({ message: "Duplicate targets in selection" })
  const selected = yield* Effect.forEach(ids, findTarget)
  const plans = selected.map(t => {
    const chosen = new Set(cases.filter(c => selection.kind === "custom" ? selection.suites.includes(c.suite)
      : selection.profile === "quick" ? quickIds.has(c.id)
      : selection.profile === "pr" ? (c.suite !== "update" || updateRepresentatives.has(t.id) && Number(c.id.slice(1)) <= 4) && (c.id !== "H7" || updateRepresentatives.has(t.id))
      : true).map(c => c.id))
    const include = (id: CaseId): void => {
      const c = cases.find(c => c.id === id)!
      for (const prerequisite of c.prerequisites) if (!chosen.has(prerequisite)) { chosen.add(prerequisite); include(prerequisite) }
    }
    for (const id of chosen) include(id)
    const expanded = cases.filter(c => chosen.has(c.id)).map(c => request.input.kind === "artifacts" && c.id === "P1"
      ? { ...c, title: "Record supplied artifact provenance without compiling" }
      : request.input.kind === "artifacts" && c.id === "P2" ? { ...c, title: "Verify supplied final package bytes without rebuilding" } : c)
      .flatMap(c => c.suite === "harness" ? harnesses.map(h => ({ ...c, harness: Option.some(h) })) : [c])
    return { target: t, cases: expanded, blockers: t.provider === "spark" && (!request.allowSpark || request.trust === "untrusted-ci")
      ? ["Spark requires explicit permission for this run and trusted source"] : [] }
  })
  // Admission uses configured rates later; this conservative reservation bounds a worst-case run.
  const execution = yield* planExecution(request, plans, targets)
  const estimate = execution.reduce((sum, work) => { const p = work.target; return sum + (p.target.provider === "spark" ? 0 : p.target.hardware === "rtx-pro-6000" ? 18 : p.target.hardware === "a10" ? 4 : p.target.provider === "namespace" ? 3.6 : 1) * request.limits.deadlineMinutes / 60 }, 0)
  return RunPlan.make({ schemaVersion: 1, request, targets: plans,
    artifactHosts: [...new Set(selected.map(t => t.artifactHost))], estimatedComputeUsd: Math.round(estimate * 100) / 100,
  })
})

export const allSuites = Suite.literals
export const allHarnesses = Harness.literals
