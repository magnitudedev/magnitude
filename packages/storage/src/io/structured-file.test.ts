import { describe, expect, it } from "vitest";
import * as FileSystem from "@effect/platform/FileSystem";
import { BunFileSystem } from "@effect/platform-bun";
import { Effect, Schema } from "effect";
import { mkdtemp } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import {
  readStructuredFile,
  readRecoverableStructuredFile,
  writeStructuredFileAtomic,
} from "./structured-file";

const run = <A, E>(effect: Effect.Effect<A, E, FileSystem.FileSystem>) =>
  Effect.runPromise(effect.pipe(Effect.provide(BunFileSystem.layer)));

const Value = Schema.Struct({ value: Schema.String });

describe("structured file", () => {
  it("distinguishes missing, invalid, and present", async () => {
    const dir = await mkdtemp(join(tmpdir(), "magnitude-structured-"));
    const path = join(dir, "value.json");

    expect((await run(readStructuredFile(path, Value)))._tag).toBe("Missing");

    await Bun.write(path, "{");
    expect((await run(readStructuredFile(path, Value)))._tag).toBe("Invalid");

    await Bun.write(path, JSON.stringify({ value: 42 }));
    expect((await run(readStructuredFile(path, Value)))._tag).toBe("Invalid");

    await run(writeStructuredFileAtomic(path, Value, { value: "complete" }));
    expect(await run(readStructuredFile(path, Value))).toEqual({
      _tag: "Present",
      value: { value: "complete" },
    });
  });

  it("leaves no temporary file after atomic publication", async () => {
    const dir = await mkdtemp(join(tmpdir(), "magnitude-structured-"));
    const path = join(dir, "value.json");

    await run(writeStructuredFileAtomic(path, Value, { value: "complete" }));

    expect(
      (await Array.fromAsync(new Bun.Glob("*.tmp").scan(dir))).length
    ).toBe(0);
  });

  it("recovers invalid leaves and escalates required failures without losing siblings", async () => {
    const dir = await mkdtemp(join(tmpdir(), "magnitude-structured-"));
    const path = join(dir, "value.json");
    const schema = Schema.Struct({
      keep: Schema.String,
      optional: Schema.optional(Schema.Number),
      nested: Schema.optional(Schema.Struct({ required: Schema.Number })),
    });
    await Bun.write(path, JSON.stringify({
      keep: "yes",
      optional: "bad",
      nested: { required: "bad" },
      future: { value: 1 },
    }));

    const result = await run(readRecoverableStructuredFile(path, schema, {
      rootDefault: () => ({ keep: "default" }),
    }));

    expect(result._tag).toBe("Present");
    if (result._tag !== "Present") return;
    expect(result.value).toEqual({ keep: "yes", future: { value: 1 } });
    expect(result.recovery.recovered).toBe(true);
    expect(result.recovery.resetRoot).toBe(false);
    expect(result.recovery.removedPaths).toContainEqual(["optional"]);
    expect(result.recovery.removedPaths).toContainEqual(["nested", "required"]);
    expect(result.recovery.removedPaths).toContainEqual(["nested"]);
  });

  it("applies a schema decoding default after removing an invalid property", async () => {
    const dir = await mkdtemp(join(tmpdir(), "magnitude-structured-"));
    const path = join(dir, "value.json");
    const schema = Schema.Struct({
      count: Schema.optionalWith(Schema.Number, { default: () => 10 }),
    });
    await Bun.write(path, JSON.stringify({ count: "bad" }));

    const result = await run(readRecoverableStructuredFile(path, schema, {
      rootDefault: () => ({ count: 10 }),
    }));

    expect(result._tag).toBe("Present");
    if (result._tag !== "Present") return;
    expect(result.value).toEqual({ count: 10 });
    expect(result.recovery.resetRoot).toBe(false);
  });

  it("removes invalid array entries without shifting later targets", async () => {
    const dir = await mkdtemp(join(tmpdir(), "magnitude-structured-"));
    const path = join(dir, "value.json");
    const schema = Schema.Struct({
      values: Schema.Array(Schema.Struct({ value: Schema.Number })),
    });
    await Bun.write(path, JSON.stringify({
      values: [{ value: "bad" }, { value: 2 }, { value: "also bad" }, { value: 4 }],
    }));

    const result = await run(readRecoverableStructuredFile(path, schema, {
      rootDefault: () => ({ values: [] }),
    }));

    expect(result._tag).toBe("Present");
    if (result._tag !== "Present") return;
    expect(result.value).toEqual({ values: [{ value: 2 }, { value: 4 }] });
  });

  it("distinguishes malformed JSON and resets an invalid root", async () => {
    const dir = await mkdtemp(join(tmpdir(), "magnitude-structured-"));
    const path = join(dir, "value.json");
    await Bun.write(path, "{");
    const malformed = await run(readRecoverableStructuredFile(path, Value, {
      rootDefault: () => ({ value: "default" }),
    }));
    expect(malformed._tag).toBe("Malformed");

    await Bun.write(path, JSON.stringify(42));
    const reset = await run(readRecoverableStructuredFile(path, Value, {
      rootDefault: () => ({ value: "default" }),
    }));
    expect(reset._tag).toBe("Present");
    if (reset._tag !== "Present") return;
    expect(reset.value).toEqual({ value: "default" });
    expect(reset.recovery.resetRoot).toBe(true);
  });

  it("preserves unknown fields through schema-aware encoding", async () => {
    const dir = await mkdtemp(join(tmpdir(), "magnitude-structured-"));
    const path = join(dir, "value.json");
    const value = { value: "complete", future: { enabled: true } };

    await run(writeStructuredFileAtomic(path, Value, value, {
      parseOptions: { onExcessProperty: "preserve" },
    }));

    expect(await Bun.file(path).json()).toEqual(value);
  });

  it("still rejects invalid application values while encoding", async () => {
    const dir = await mkdtemp(join(tmpdir(), "magnitude-structured-"));
    const path = join(dir, "value.json");

    const result = await run(Effect.either(
      writeStructuredFileAtomic(path, Value, { value: 42 } as never)
    ));

    expect(result._tag).toBe("Left");
    if (result._tag === "Left") expect(result.left._tag).toBe("StructuredFileEncodeFailed");
    expect(await Bun.file(path).exists()).toBe(false);
  });
});
