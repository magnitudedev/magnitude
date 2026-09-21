import { Effect, Option, Schema } from "effect"
import { Backend, InvalidInput, RunRequest, Target, TargetId, TargetPlan } from "./domain"

import { WorkId } from "./work-identity"
export { WorkId } from "./work-identity"
export const BuildWork = Schema.Struct({ kind: Schema.Literal("build"), id: WorkId, target: TargetPlan,
  backend: Backend, consumers: Schema.NonEmptyArray(TargetId) })
export const TestWork = Schema.Struct({ kind: Schema.Literal("test"), id: WorkId, target: TargetPlan,
  producer: Schema.optionalWith(WorkId, { as: "Option", exact: true }) })
export const WorkSpec = Schema.Union(BuildWork, TestWork)
export type WorkSpec = typeof WorkSpec.Type
export const ExecutionPlan = Schema.Array(WorkSpec)

const buildCases = new Set(["P1", "P2"])
const producerTargets: Readonly<Record<Target["artifactHost"], string>> = {
  "darwin-arm64": "macos-15-arm64-cpu-apple-silicon",
  "windows-x64-msvc": "windows-server-2022-x64-cpu-intel",
  "linux-x64-gnu": "ubuntu-24.04-x64-cpu-intel",
  "linux-arm64-gnu": "ubuntu-24.04-arm64-cpu-arm",
}

/** Build identity is scoped to one immutable run; hardware and distro remain consumer identities. */
export const planExecution = (request: RunRequest, consumers: readonly (typeof TargetPlan.Type)[], catalog: readonly Target[]) => Effect.gen(function* () {
  if (request.input.kind === "artifacts") return consumers.map(target => TestWork.make({ kind: "test",
    id: WorkId.make(`test:${target.target.id}`), target, producer: Option.none() }))
  const groups = new Map<WorkId, (typeof TargetPlan.Type)[]>()
  const producerId = (target: Target) => WorkId.make(`build:${target.artifactHost}:${target.backend}`)
  for (const consumer of consumers) {
    const id = producerId(consumer.target)
    const group = groups.get(id) ?? []
    group.push(consumer)
    groups.set(id, group)
  }
  const builds = yield* Effect.forEach([...groups.entries()].sort(([left], [right]) => left.localeCompare(right)), ([id, group]) => Effect.gen(function* () {
    const representative = group[0]!
    const machine = catalog.find(target => target.id === producerTargets[representative.target.artifactHost])
    if (!machine) return yield* new InvalidInput({ message: `No native build machine for ${representative.target.artifactHost}` })
    const cases = representative.cases.filter(test => buildCases.has(test.id))
    if (cases.length !== 2 || cases.some(test => Option.isSome(test.harness))) return yield* new InvalidInput({ message: "Source execution requires shared compilation and packaging cases" })
    const ids = group.map(consumer => consumer.target.id).sort()
    const target = TargetPlan.make({ target: machine, cases,
      blockers: group.every(consumer => consumer.blockers.length > 0) ? ["All consumers of this build are blocked"] : [] })
    return BuildWork.make({ kind: "build", id, target, backend: representative.target.backend, consumers: [ids[0]!, ...ids.slice(1)] })
  }))
  const tests = consumers.map(consumer => TestWork.make({ kind: "test", id: WorkId.make(`test:${consumer.target.id}`),
    producer: Option.some(producerId(consumer.target)), target: { ...consumer,
      // Durable dependency admission supplies P1/P2; consumers never compile or package.
      cases: consumer.cases.filter(test => !buildCases.has(test.id)).map(test => ({ ...test,
        prerequisites: test.prerequisites.filter(id => !buildCases.has(id)),
      })),
    } }))
  return [...builds, ...tests]
})
