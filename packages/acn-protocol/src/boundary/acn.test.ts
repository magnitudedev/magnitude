import { Context } from "effect"
import { describe, expect, expectTypeOf, it } from "vitest"
import { RpcSchema, Rpc as RpcDefinition } from "@effect/rpc"
import { rpcGroup, type TreeRpcs } from "../rpc-tree"
import type * as Rpc from "@effect/rpc/Rpc"
import { AcnRpcRecoveryPolicyTag } from "../transport/recovery"
import { AcnRpcGroup, MagnitudeRpcs } from "./acn"
import { Connection } from "./connection"
import { Sessions } from "./sessions"

describe("ACN boundary lifecycle policy", () => {
  const healthTag: Extract<TreeRpcs<typeof MagnitudeRpcs>["_tag"], "Health"> = "Health"
  void healthTag
  type HealthRpc = Extract<TreeRpcs<typeof Connection>, { readonly _tag: "Health" }>
  type SessionQueryRpc = Extract<TreeRpcs<typeof Sessions>, { readonly _tag: "ListSessions" }>
  expectTypeOf<Rpc.Middleware<HealthRpc>>().toEqualTypeOf<never>()
  expectTypeOf<Rpc.Middleware<SessionQueryRpc>>().toEqualTypeOf<never>()
  it("derives finite recovery policy without lifecycle middleware", () => {
    for (const operation of AcnRpcGroup.requests.values()) {
      const recovery = Context.getOption(operation.annotations, AcnRpcRecoveryPolicyTag)

      if (RpcSchema.isStreamSchema(operation.successSchema)) {
        expect(recovery._tag).toBe("None")
      } else {
        expect(recovery._tag).toBe("Some")
      }
    }
  })

  it("derives exactly the ACN stream operations", () => {
    const streams = [...AcnRpcGroup.requests.values()]
      .filter((operation) => RpcSchema.isStreamSchema(operation.successSchema))
      .map((operation) => operation._tag)
      .sort()
    expect(streams).toEqual([
      "StreamActiveSessionStatuses",
      "StreamChanges",
      "StreamDisplayView",
      "WatchFile",
      "WatchProjectFiles",
    ])
  })

  it("rejects duplicate tags and malformed namespaces", () => {
    expect(() => rpcGroup({ a: Connection.health, b: Connection.health })).toThrow("Duplicate RPC tag")
    expect(() => rpcGroup({ a: null } as never)).toThrow("Invalid RPC namespace")
  })

  it("rejects an undeclared finite replay policy at group construction", () => {
    // @ts-expect-error A finite RPC cannot enter the tree without an explicit recovery declaration.
    expect(() => rpcGroup({ missing: RpcDefinition.make("MissingPolicy") })).toThrow("Missing RPC recovery policy")
  })
})
