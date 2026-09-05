import { describe, expect, it } from "vitest";
import { Effect, Schema } from "effect";
import { Rpc, RpcGroup } from "@effect/rpc";
import { AcnRpcRecoveryPolicyTag } from "@magnitudedev/acn-protocol";
import { canonical } from "@magnitudedev/utils/canonical-key";
import { describeWireSchema, rpcFingerprint } from "./rpc-fingerprint";

const describeSchema = (schema: Schema.Schema.All) =>
  Effect.runPromise(describeWireSchema(schema));
describe("RPC wire fingerprint", () => {
  it("ignores property order, brands, decoded classes, and descriptions", async () => {
    const a = Schema.Struct({ a: Schema.String, b: Schema.Number }).annotations(
      { description: "first" }
    );
    const b = Schema.Struct({
      b: Schema.Number,
      a: Schema.String.pipe(Schema.brand("Label")),
    });
    expect(canonical(await describeSchema(a))).toBe(
      canonical(await describeSchema(b))
    );
    class Value extends Schema.Class<Value>("Value")({
      a: Schema.String,
      b: Schema.Number,
    }) {}
    expect(canonical(await describeSchema(Value))).toBe(
      canonical(await describeSchema(a))
    );
  });
  it("detects optionality, encoded transformations, bounds, tuples, and unions", async () => {
    for (const [a, b] of [
      [
        Schema.Struct({ x: Schema.String }),
        Schema.Struct({
          x: Schema.optionalWith(Schema.String, { exact: true, as: "Option" }),
        }),
      ],
      [Schema.Number, Schema.NumberFromString],
      [
        Schema.Number.pipe(Schema.greaterThan(0)),
        Schema.Number.pipe(Schema.greaterThan(1)),
      ],
      [Schema.Tuple(Schema.String), Schema.Array(Schema.String)],
      [Schema.Literal("a"), Schema.Literal("a", "b")],
    ] as const)
      expect(canonical(await describeSchema(a))).not.toBe(
        canonical(await describeSchema(b))
      );
  });
  it("handles recursive JSON and exposes opaque predicates for semantic review", async () => {
    interface Node {
      readonly next?: Node;
    }
    const node: Schema.Schema<Node> = Schema.Struct({
      next: Schema.optional(Schema.suspend(() => node)),
    });
    expect(JSON.stringify(await describeSchema(node))).toContain("recursive");
    const opaque = await describeSchema(
      Schema.String.pipe(Schema.filter((value) => value !== "forbidden"))
    );
    expect(opaque).toMatchObject({ kind: "refinement", opaquePredicate: true });
    await expect(describeSchema(Schema.DateFromSelf)).rejects.toThrow(
      "Declaration"
    );
  });
  it("detects procedure, stream and replay changes without depending on namespaces", async () => {
    const rpc = Rpc.make("Read", {
      payload: {},
      success: Schema.String,
    }).annotate(AcnRpcRecoveryPolicyTag, "ReplaySafe");
    const fingerprint = (r: Rpc.Any & Rpc.AnyWithProps) =>
      Effect.runPromise(rpcFingerprint(RpcGroup.make(r)));
    const before = await fingerprint(rpc);
    expect(
      await fingerprint(rpc.annotate(AcnRpcRecoveryPolicyTag, "AtMostOnce"))
    ).not.toBe(before);
    expect(
      await fingerprint(
        Rpc.make("Read", { payload: {}, success: Schema.String, stream: true })
      )
    ).not.toBe(before);
  });
  it("describes every current procedure deterministically", async () => {
    const first = await Effect.runPromise(rpcFingerprint());
    const second = await Effect.runPromise(rpcFingerprint());
    expect(first).toBe(second);
    expect(first).toMatch(/^[a-f0-9]{64}$/);
  });
  it("detects an encoded result change", async () => {
    const group = (value: string) =>
      RpcGroup.make(
        Rpc.make("Read", { success: Schema.Literal(value) }).annotate(
          AcnRpcRecoveryPolicyTag,
          "ReplaySafe"
        )
      );
    const before = await Effect.runPromise(rpcFingerprint(group("old")));
    const after = await Effect.runPromise(rpcFingerprint(group("new")));
    expect(after).not.toBe(before);
  });
});
