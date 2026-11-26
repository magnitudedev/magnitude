import { Context } from "effect"
import { describe, expect, expectTypeOf, it } from "vitest"
import { RpcSchema, Rpc as RpcDefinition } from "@effect/rpc"
import { rpcGroup, type TreeRpcs } from "../rpc-tree"
import type * as Rpc from "@effect/rpc/Rpc"
import { AcnRpcRecoveryPolicyTag } from "../transport/recovery"
import { AcnRpcGroup, MagnitudeRpcs, RemoteAcnRpcGroup } from "./acn"
import { Connection } from "./connection"
import { Configuration } from "./configuration"

describe("ACN boundary lifecycle policy", () => {
  const healthTag: Extract<TreeRpcs<typeof MagnitudeRpcs>["_tag"], "Health"> = "Health"
  void healthTag
  type HealthRpc = Extract<TreeRpcs<typeof Connection>, { readonly _tag: "Health" }>
  type StorageQueryRpc = Extract<TreeRpcs<typeof Configuration>, { readonly _tag: "GetModelStorage" }>
  expectTypeOf<Rpc.Middleware<HealthRpc>>().toEqualTypeOf<never>()
  expectTypeOf<Rpc.Middleware<StorageQueryRpc>>().toEqualTypeOf<never>()
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
      "StreamChanges",
      "StreamModelCandidateInspection",
      "WatchApplicationOwner",
      "WatchChat",
      "WatchHarnessConnections",
    ])
  })

  it("keeps RemoteAcnRpcGroup a strict subset of AcnRpcGroup excluding loopback-only operations", () => {
    const localTags = new Set(AcnRpcGroup.requests.keys())
    const remoteTags = new Set(RemoteAcnRpcGroup.requests.keys())
    for (const tag of remoteTags) {
      expect(localTags.has(tag)).toBe(true)
    }
    for (const loopbackOnly of [
      "SetNetworkAccess",
      "RegenerateNetworkApiKey",
      "WatchHarnessConnections",
      "ConnectHarness",
      "SyncHarnessConnections",
      "DisconnectHarness",
      "StartHuggingFaceSignIn",
      "GetServiceActivity",
      "DownloadApplicationUpdate",
      "DiscardApplicationUpdate",
      "InstallApplicationUpdate",
      "SetApplicationAutoDownload",
      "SetLaunchAtLogin",
      "QuitApplication",
    ]) {
      expect(localTags.has(loopbackOnly)).toBe(true)
      expect(remoteTags.has(loopbackOnly)).toBe(false)
    }
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
