import { describe, expect, it } from "vitest";
import * as FileSystem from "@effect/platform/FileSystem";
import { BunFileSystem } from "@effect/platform-bun";
import { Effect, Schema } from "effect";
import { mkdtemp } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import {
  readStructuredFile,
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
});
