import {
  HttpServerRequest,
  HttpServerResponse,
} from "@effect/platform"
import { describe, expect, it } from "vitest"
import { Effect, Option } from "effect"
import { makeAcnStartupState } from "./startup-state"

describe("ACN startup state", () => {
  it("changes health and RPC admission through one state transition", async () => {
    const result = await Effect.runPromise(
      Effect.scoped(Effect.gen(function* () {
        const startup = yield* makeAcnStartupState()
        const startingResponse = yield* startup.rpc
        yield* startup.ready(
          Effect.succeed(HttpServerResponse.empty({ status: 204 })),
        )
        const readyHealth = yield* startup.get
        const readyResponse = yield* startup.rpc
        yield* startup.starting("Resolving", Option.none())
        const restartedResponse = yield* startup.rpc
        return {
          startingStatus: startingResponse.status,
          readyHealth,
          readyStatus: readyResponse.status,
          restartedStatus: restartedResponse.status,
        }
      })).pipe(
        Effect.provideService(
          HttpServerRequest.HttpServerRequest,
          HttpServerRequest.fromWeb(
            new Request("http://127.0.0.1/rpc", { method: "POST" }),
          ),
        ),
      ),
    )

    expect(result).toEqual({
      startingStatus: 503,
      readyHealth: { _tag: "Ready" },
      readyStatus: 204,
      restartedStatus: 503,
    })
  })
})
