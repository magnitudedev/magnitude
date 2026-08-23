import { Context } from "effect"
import { describe, expect, expectTypeOf, it } from "vitest"
import type { GroupRpcs } from "@magnitudedev/effect-query/rpc"
import type * as Rpc from "@effect/rpc/Rpc"
import { AcnRpcRecoveryPolicyTag } from "../transport/recovery"
import { AcnBoundary } from "./acn"
import { AcnRpc } from "./rpc"
import { Connection } from "./connection"
import { Sessions } from "./sessions"

describe("ACN boundary lifecycle policy", () => {
  const healthTag: Extract<GroupRpcs<typeof AcnBoundary>["_tag"], "Health"> = "Health"
  void healthTag
  type HealthRpc = Extract<GroupRpcs<typeof Connection>, { readonly _tag: "Health" }>
  type SessionQueryRpc = Extract<GroupRpcs<typeof Sessions>, { readonly _tag: "ListSessions" }>
  expectTypeOf<Rpc.Middleware<HealthRpc>>().toEqualTypeOf<never>()
  expectTypeOf<Rpc.Middleware<SessionQueryRpc>>().toEqualTypeOf<never>()
  it("derives finite recovery policy without lifecycle middleware", () => {
    for (const operation of AcnRpc.operations(AcnBoundary)) {
      const recovery = Context.getOption(operation.annotations, AcnRpcRecoveryPolicyTag)

      if (operation.stream) {
        expect(recovery._tag).toBe("None")
      } else {
        expect(recovery._tag).toBe("Some")
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
