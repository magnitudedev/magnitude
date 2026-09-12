import { describe, expect, it } from "vitest"
import { RpcSchema } from "@effect/rpc"
import { Context, Option } from "effect"
import { AcnRpcRecoveryPolicyTag } from "@magnitudedev/sdk"
import { InferenceHostRpcs } from "./desktop-rpc"

describe("desktop host recovery contract", () => {
  it("declares every finite call and never allows host actions to be replayed", () => {
    for (const rpc of InferenceHostRpcs.requests.values()) {
      if (RpcSchema.isStreamSchema(rpc.successSchema)) continue
      expect(Context.getOption(rpc.annotations, AcnRpcRecoveryPolicyTag), rpc._tag)
        .toEqual(Option.some(rpc._tag === "ApplicationInfo" ? "ReplaySafe" : "AtMostOnce"))
    }
  })
})
