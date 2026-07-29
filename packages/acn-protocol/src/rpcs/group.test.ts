import { describe, expect, it } from "vitest"
import type { RpcMiddleware } from "@effect/rpc"
import { Context, Option } from "effect"
import { MagnitudeRpcs } from "./group"
import { AcnRpcDemand } from "./middleware"
import { AcnSubscriptionMetadataTag } from "./subscription"

const hasDemandMiddleware = (
  middlewares: ReadonlySet<RpcMiddleware.TagClassAny>,
): boolean => middlewares.has(AcnRpcDemand)

describe("ACN RPC lifecycle policy", () => {
  it("derives one unambiguous policy for every RPC", () => {
    for (const [tag, rpc] of MagnitudeRpcs.requests) {
      const subscription = Option.isSome(
        Context.getOption(rpc.annotations, AcnSubscriptionMetadataTag),
      )
      const demand = hasDemandMiddleware(rpc.middlewares)

      if (tag === "Health") {
        expect({ subscription, demand }).toEqual({ subscription: false, demand: false })
      } else if (subscription) {
        expect(demand).toBe(false)
      } else {
        expect(demand).toBe(true)
      }
    }
  })
})
