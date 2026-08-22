import { Context } from "effect"
import { describe, expect, expectTypeOf, it } from "vitest"
import type { GroupRpcs } from "@magnitudedev/effect-query/rpc"
import type * as Rpc from "@effect/rpc/Rpc"
import { AcnRpcDemandPolicyTag } from "../transport/demand"
import { AcnRpcDemand } from "../transport/demand"
import { AcnRpcRecoveryPolicyTag } from "../transport/recovery"
import { AcnBoundary } from "./acn"
import { AcnRpc } from "./rpc"
import { Connection } from "./connection"
import { Sessions } from "./sessions"

describe("ACN boundary lifecycle policy", () => {
  const healthTag: Extract<GroupRpcs<typeof AcnBoundary, typeof AcnRpcDemand>["_tag"], "Health"> = "Health"
  void healthTag
  type HealthRpc = Extract<GroupRpcs<typeof Connection, typeof AcnRpcDemand>, { readonly _tag: "Health" }>
  type SessionQueryRpc = Extract<GroupRpcs<typeof Sessions, typeof AcnRpcDemand>, { readonly _tag: "ListSessions" }>
  expectTypeOf<Rpc.Middleware<HealthRpc>>().toEqualTypeOf<AcnRpcDemand>()
  expectTypeOf<Rpc.Middleware<SessionQueryRpc>>().toEqualTypeOf<AcnRpcDemand>()
  it("derives one unambiguous policy for every operation", () => {
    for (const operation of AcnRpc.operations(AcnBoundary)) {
      const demand = Context.getOption(operation.annotations, AcnRpcDemandPolicyTag)
      const recovery = Context.getOption(operation.annotations, AcnRpcRecoveryPolicyTag)

      if (operation.stream) {
        expect(demand._tag).toBe("None")
        expect(recovery._tag).toBe("None")
      } else {
        expect(recovery._tag).toBe("Some")
        expect(demand._tag).toBe("Some")
        expect(demand._tag === "Some" && demand.value).toBe(
          operation.name !== "Health"
            && operation.name !== "RenewClientLease"
            && operation.name !== "ReleaseClientLease",
        )
      }
    }
  })

  it("derives exactly the ACN stream operations", () => {
    const streams = AcnRpc.operations(AcnBoundary)
      .filter((operation) => operation.stream)
      .map((operation) => operation.name)
      .sort()
    expect(streams).toEqual([
      "StreamActiveSessionStatuses",
      "StreamChanges",
      "StreamDisplayView",
      "WatchFile",
      "WatchProjectFiles",
    ])
  })
})
