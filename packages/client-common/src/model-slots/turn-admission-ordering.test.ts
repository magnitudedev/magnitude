import { Registry } from "@effect-atom/atom-react"
import { Deferred, Effect, Fiber, Option } from "effect"
import { describe, expect, it } from "vitest"
import { Client, Mutation } from "@magnitudedev/effect-query"
import {
  AcnBoundary,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  type ModelSlotsState,
} from "@magnitudedev/sdk"
import { clientServicesLayer, type ClientServices } from "../state/client-services"
import type { AcnClientRequirements } from "../state/agent-client"
import { fakeAcnImplementationsLayer } from "../state/fake-acn-implementations"

const selection = {
  providerId: ProviderIdSchema.make("local"),
  providerModelId: ProviderModelIdSchema.make("new-model"),
  reasoningEffort: ReasoningEffortSchema.make("high"),
}

const modelSlots = {
  slots: {
    primary: {
      _tag: "ConfiguredLocal",
      slotId: PRIMARY_SLOT_ID,
      selection,
      descriptor: {
        providerId: selection.providerId,
        providerModelId: selection.providerModelId,
        displayName: "New model",
        variantLabel: Option.none(),
      },
      availability: { _tag: "Available" },
      residency: { _tag: "Unloaded" },
      actions: ["Load"],
    },
    secondary: { _tag: "Unassigned", slotId: "secondary" },
  },
  recentModels: { primary: [], secondary: [] },
  favoriteModels: [],
} as unknown as ModelSlotsState

const reproduceOvertake = (submit: (client: ReturnType<typeof makeClient>) =>
  Effect.Effect<unknown, unknown, Registry.AtomRegistry>) =>
  Effect.scoped(Effect.gen(function* () {
    const assignmentSynchronizationEntered = yield* Deferred.make<void>()
    const releaseAssignmentSynchronization = yield* Deferred.make<void>()
    const turnMutationEntered = yield* Deferred.make<void>()
    let assignmentCommitted = false

    const implementation = fakeAcnImplementationsLayer((name) => {
        if (name === "AssignSlot") {
          assignmentCommitted = true
          return Effect.succeed({})
        }
        if (name === "GetModelSlots") {
          return Effect.gen(function* () {
            if (assignmentCommitted) {
              yield* Deferred.succeed(assignmentSynchronizationEntered, undefined)
              yield* Deferred.await(releaseAssignmentSynchronization)
            }
            return { revision: 1, state: modelSlots }
          })
        }
        if (name === "SendMessage" || name === "StartGoal" || name === "CreateSession") {
          return Deferred.succeed(turnMutationEntered, undefined).pipe(Effect.as({}))
        }
        return Effect.dieMessage(`Unexpected operation ${name}`)
    })
    const client = makeClient(implementation)
    const registry = Registry.make()

    const assignment = yield* Mutation.execute(client.Configuration.AssignSlot, {
      slotId: PRIMARY_SLOT_ID,
      selection,
    }).pipe(
      Effect.provideService(Registry.AtomRegistry, registry),
      Effect.fork,
    )
    yield* Deferred.await(assignmentSynchronizationEntered)

    const submission = yield* submit(client).pipe(
      Effect.provideService(Registry.AtomRegistry, registry),
      Effect.fork,
    )
    yield* Effect.yieldNow()
    const overtookSynchronization = Option.isSome(yield* Deferred.poll(turnMutationEntered))

    yield* Deferred.succeed(releaseAssignmentSynchronization, undefined)
    yield* Fiber.join(assignment)
    yield* Fiber.join(submission)
    registry.dispose()

    return overtookSynchronization
  }))

const makeClient = (implementation: ReturnType<typeof fakeAcnImplementationsLayer>) =>
  Client.make<typeof AcnBoundary, AcnClientRequirements, never, ClientServices, never>(
    AcnBoundary,
    implementation,
    clientServicesLayer,
  )

describe("turn admission ordering after primary model selection", () => {
  it("does not let SendMessage overtake AssignSlot synchronization", async () => {
    const overtook = await Effect.runPromise(reproduceOvertake((client) =>
      Mutation.execute(client.Agent.SendMessage, {
        sessionId: "session",
        messageId: Option.some("message"),
        content: "hello",
        visibleMessage: Option.none(),
        taskMode: false,
        uploads: [],
        mentions: [],
      })))
    expect(overtook).toBe(false)
  })

  it("does not let StartGoal overtake AssignSlot synchronization", async () => {
    const overtook = await Effect.runPromise(reproduceOvertake((client) =>
      Mutation.execute(client.Agent.StartGoal, {
        sessionId: "session",
        objective: "goal",
      })))
    expect(overtook).toBe(false)
  })

  it("does not let initial session work overtake AssignSlot synchronization", async () => {
    const overtook = await Effect.runPromise(reproduceOvertake((client) =>
      Mutation.execute(client.Sessions.CreateSession, {
        cwd: "/project",
        sessionId: Option.none(),
        initial: Option.some({
          _tag: "message" as const,
          messageId: Option.some("message"),
          content: "hello",
          visibleMessage: Option.none<string>(),
          taskMode: false,
          uploads: [],
          mentions: [],
        }),
        options: Option.none(),
        draftOwnerId: Option.none(),
      })))
    expect(overtook).toBe(false)
  })
})
