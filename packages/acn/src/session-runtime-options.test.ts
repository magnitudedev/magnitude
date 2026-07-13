import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { BunFileSystem, BunPath } from "@effect/platform-bun";
import {
  GlobalStorage,
  ProjectStorage,
  StorageLive,
  Version,
  makeGlobalStoragePaths,
  makeProjectStoragePaths,
} from "@magnitudedev/storage";
import { Effect, Layer } from "effect";
import { afterEach, beforeEach, describe, expect, test } from "vitest";
import {
  normalizeSessionRuntimeOptions,
  SessionRuntimeOptionsStore,
  SessionRuntimeOptionsStoreLive,
} from "./session-runtime-options";

const VERSION = "0.0.1";
const sessionId = "session-runtime-options-test";

function makeTestLayer(root: string) {
  const base = Layer.mergeAll(
    BunFileSystem.layer,
    BunPath.layer,
    Layer.succeed(Version, Version.of({ getVersion: () => VERSION })),
    Layer.succeed(
      GlobalStorage,
      GlobalStorage.of({
        root,
        paths: makeGlobalStoragePaths(root),
      })
    ),
    Layer.succeed(
      ProjectStorage,
      ProjectStorage.of({
        cwd: "/repo",
        root: join(root, "project"),
        paths: makeProjectStoragePaths(root),
      })
    )
  );
  const storageLayer = StorageLive.pipe(Layer.provide(base));
  return SessionRuntimeOptionsStoreLive.pipe(
    Layer.provide(Layer.mergeAll(storageLayer, BunFileSystem.layer))
  );
}

const run = <A, E>(
  eff: Effect.Effect<A, E, SessionRuntimeOptionsStore>,
  root: string
) => Effect.runPromise(eff.pipe(Effect.provide(makeTestLayer(root))));

describe("SessionRuntimeOptionsStore", () => {
  let tmpDir: string;
  let paths: ReturnType<typeof makeGlobalStoragePaths>;

  const optionsPath = () =>
    join(paths.sessionDir(sessionId), "runtime-options.json");

  beforeEach(async () => {
    tmpDir = await mkdtemp(
      join(tmpdir(), "magnitude-acn-session-runtime-options-")
    );
    paths = makeGlobalStoragePaths(tmpDir);
  });

  afterEach(async () => {
    await rm(tmpDir, { recursive: true, force: true });
  });

  test("normalizes absent protocol options to total runtime defaults", () => {
    expect(normalizeSessionRuntimeOptions()).toEqual({
      disableShellSafeguards: false,
      disableCwdSafeguards: false,
      atifPath: null,
      solo: false,
      systemPromptOverride: null,
      headless: false,
    });
  });

  test("normalizes provided options, defaulting unset fields", () => {
    expect(
      normalizeSessionRuntimeOptions({
        disableShellSafeguards: true,
        solo: true,
      })
    ).toEqual({
      disableShellSafeguards: true,
      disableCwdSafeguards: false,
      atifPath: null,
      solo: true,
      systemPromptOverride: null,
      headless: false,
    });
  });

  test("writes then reads the options round-trip", async () => {
    const options = normalizeSessionRuntimeOptions({
      disableShellSafeguards: true,
      disableCwdSafeguards: true,
      solo: true,
      headless: true,
    });

    const result = await run(
      Effect.gen(function* () {
        const store = yield* SessionRuntimeOptionsStore;
        yield* store.write(sessionId, options);
        return yield* store.read(sessionId);
      }),
      tmpDir
    );

    expect(result).toEqual(options);
  });

  test("writes plain JSON — no envelope, no schema version, no source", async () => {
    const options = normalizeSessionRuntimeOptions({ solo: true });

    await run(
      Effect.gen(function* () {
        const store = yield* SessionRuntimeOptionsStore;
        yield* store.write(sessionId, options);
      }),
      tmpDir
    );

    const raw = JSON.parse(await readFile(optionsPath(), "utf8"));
    expect(raw).toEqual(options);
    expect(raw).not.toHaveProperty("schemaVersion");
    expect(raw).not.toHaveProperty("source");
    expect(raw).not.toHaveProperty("updatedAt");
    expect(raw).not.toHaveProperty("sessionId");
  });

  test("read returns null when the file is missing", async () => {
    const result = await run(
      Effect.gen(function* () {
        const store = yield* SessionRuntimeOptionsStore;
        return yield* store.read(sessionId);
      }),
      tmpDir
    );

    expect(result).toBeNull();
  });

  test("read returns a typed failure when the durable file is corrupt", async () => {
    await mkdir(paths.sessionDir(sessionId), { recursive: true });
    await writeFile(optionsPath(), "{ not json", "utf8");

    const result = await run(
      Effect.exit(
        Effect.gen(function* () {
          const store = yield* SessionRuntimeOptionsStore;
          return yield* store.read(sessionId);
        })
      ),
      tmpDir
    );

    expect(result._tag).toBe("Failure");
  });

  test("read does not write when the file is missing", async () => {
    await run(
      Effect.gen(function* () {
        const store = yield* SessionRuntimeOptionsStore;
        return yield* store.read(sessionId);
      }),
      tmpDir
    );

    // File should not have been created by the read
    await expect(readFile(optionsPath(), "utf8")).rejects.toThrow();
  });
});
