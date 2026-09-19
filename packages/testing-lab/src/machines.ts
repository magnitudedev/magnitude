import { Context, Effect, Schema } from "effect"
import { Allocating } from "./lease"
import { InfrastructureFailure, LeaseId, RunId, Target } from "./domain"

export const MachineTags = Schema.Struct({ schemaVersion: Schema.Literal(1), runId: RunId, leaseId: LeaseId, expiresAt: Schema.DateTimeUtc })
export type MachineTags = typeof MachineTags.Type
export const NamespaceMachine = Schema.Struct({ provider: Schema.Literal("namespace"), id: Schema.NonEmptyString, name: Schema.NonEmptyString, tags: MachineTags })
export const AzureMachine = Schema.Struct({ provider: Schema.Literal("azure"), id: Schema.NonEmptyString, name: Schema.NonEmptyString, tags: MachineTags })
export const SshMachine = Schema.Struct({ provider: Schema.Literal("spark"), host: Schema.NonEmptyString, tags: MachineTags })
export const LocalMachine = Schema.Struct({ provider: Schema.Literal("local"), root: Schema.NonEmptyString, tags: MachineTags })
export const Machine = Schema.Union(NamespaceMachine, AzureMachine, SshMachine, LocalMachine)
export type Machine = typeof Machine.Type
export interface MachineAllocator {
  readonly ensure: (lease: Allocating, target: Target) => Effect.Effect<Machine, InfrastructureFailure>
  readonly release: (machine: Machine) => Effect.Effect<void, InfrastructureFailure>
  readonly inventory: () => Effect.Effect<ReadonlyArray<Machine>, InfrastructureFailure>
}
export const MachineAllocator = Context.GenericTag<MachineAllocator>("@magnitudedev/testing-lab/MachineAllocator")

export interface WorkerTransport {
  readonly execute: (machine: Machine, executable: string, args: readonly string[], timeoutMs: number) => Effect.Effect<{ readonly stdout: string; readonly stderr: string; readonly exitCode: number }, InfrastructureFailure>
  readonly upload: (machine: Machine, local: string, remote: string) => Effect.Effect<void, InfrastructureFailure>
  readonly download: (machine: Machine, remote: string, local: string) => Effect.Effect<void, InfrastructureFailure>
}
export const WorkerTransport = Context.GenericTag<WorkerTransport>("@magnitudedev/testing-lab/WorkerTransport")
