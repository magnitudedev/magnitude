import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { describe, expect, test, beforeEach, afterEach } from "vitest";
import { Effect, Layer } from "effect";
import { BunFileSystem, BunPath } from "@effect/platform-bun";

import { makeGlobalStoragePaths, makeProjectStoragePaths } from "../paths";
import { GlobalStorage } from "../services/global-storage";
import { ProjectStorage } from "../services/project-storage";
import { Version } from "../services/version";
import { MagnitudeStorage, StorageLive } from "../storage";

const VERSION = "0.0.1";

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
  return StorageLive.pipe(Layer.provide(base));
}

const testLayer = (root: string) => makeTestLayer(root);

const run = <A, E>(eff: Effect.Effect<A, E, MagnitudeStorage>, root: string) =>
  Effect.runPromise(eff.pipe(Effect.provide(testLayer(root))));

describe("session storage", () => {
  let tmpDir: string;
  let paths: ReturnType<typeof makeGlobalStoragePaths>;
  const sessionId = "session-1";

  beforeEach(async () => {
    tmpDir = await mkdtemp(join(tmpdir(), "magnitude-storage-sessions-"));
    paths = makeGlobalStoragePaths(tmpDir);
  });

  afterEach(async () => {
    await rm(tmpDir, { recursive: true, force: true });
  });

  const baseMeta = {
    sessionId,
    projectId: "project-1",
    sidebarOpen: true,
    chatName: "Chat",
    workingDirectory: "/repo",
    visibility: "visible",
    created: "2026-01-01T00:00:00.000Z",
    updated: "2026-01-01T00:00:00.000Z",
    initialVersion: "0.0.1",
    lastActiveVersion: "0.0.1",
    gitBranch: null,
    firstUserMessage: null,
    lastMessage: null,
    messageCount: 0,
  };

  test("readMeta returns a typed failure for corrupt meta.json", async () => {
    await mkdir(paths.sessionDir(sessionId), { recursive: true });
    await writeFile(
      paths.sessionMetaFile(sessionId),
      "{ not valid json",
      "utf-8"
    );

    const exit = await Effect.runPromise(
      Effect.exit(
        Effect.gen(function* () {
          const storage = yield* MagnitudeStorage;
          return yield* storage.sessions.readMeta(sessionId);
        }).pipe(Effect.provide(testLayer(tmpDir)))
      )
    );

    expect(exit._tag).toBe("Failure");
  });

  test("writeMeta writes and reads back", async () => {
    await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage;
        yield* storage.sessions.writeMeta(sessionId, baseMeta as any);
      }),
      tmpDir
    );

    const result = await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage;
        return yield* storage.sessions.readMeta(sessionId);
      }),
      tmpDir
    );
    expect(result).toEqual(baseMeta);
  });

  test("updateMeta updates", async () => {
    await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage;
        yield* storage.sessions.writeMeta(sessionId, baseMeta as any);
        yield* storage.sessions.updateMeta(
          sessionId,
          () =>
            ({
              ...baseMeta,
              chatName: "Updated",
              updated: "2026-01-02T00:00:00.000Z",
            } as any)
        );
      }),
      tmpDir
    );

    const result = await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage;
        return yield* storage.sessions.readMeta(sessionId);
      }),
      tmpDir
    );
    expect(result).toEqual({
      ...baseMeta,
      chatName: "Updated",
      updated: "2026-01-02T00:00:00.000Z",
    });
  });

  test("serializes concurrent metadata updates without losing writes", async () => {
    const result = await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage;
        yield* storage.sessions.writeMeta(sessionId, baseMeta as any);
        yield* Effect.all(
          Array.from({ length: 20 }, () =>
            storage.sessions.updateMeta(
              sessionId,
              (current) =>
                ({
                  ...current!,
                  messageCount: current!.messageCount + 1,
                } as any)
            )
          ),
          { concurrency: "unbounded" }
        );
        return yield* storage.sessions.readMeta(sessionId);
      }),
      tmpDir
    );

    expect(result?.messageCount).toBe(20);
  });

  test("writeMeta prepends new session ids and updateMeta preserves existing index order", async () => {
    await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage;
        yield* storage.sessions.writeMeta("mqa00000", {
          ...baseMeta,
          sessionId: "mqa00000",
          updated: "2026-01-01T00:00:00.000Z",
        } as any);
        yield* storage.sessions.writeMeta("mqa00001", {
          ...baseMeta,
          sessionId: "mqa00001",
          updated: "2026-01-02T00:00:00.000Z",
        } as any);
        yield* storage.sessions.updateMeta(
          "mqa00000",
          (current) =>
            ({
              ...current!,
              updated: "2026-01-03T00:00:00.000Z",
            } as any)
        );
        return yield* storage.sessions.readCwdIndex("/repo");
      }),
      tmpDir
    );

    const index = await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage;
        return yield* storage.sessions.readCwdIndex("/repo");
      }),
      tmpDir
    );

    expect(index?.sessionIds).toEqual(["mqa00001", "mqa00000"]);
  });

  test("event appends preserve order", async () => {
    await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage;
        yield* storage.sessions.appendEventsWithCursor(sessionId, [
          { type: "a", timestamp: 100 },
        ]);
        yield* storage.sessions.appendEventsWithCursor(sessionId, [
          { type: "b", timestamp: 200 },
        ]);
      }),
      tmpDir
    );

    const result = await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage;
        return yield* storage.sessions.readEvents<{ type: string }>(sessionId);
      }),
      tmpDir
    );
    expect(result).toEqual([
      { type: "a", timestamp: 100 },
      { type: "b", timestamp: 200 },
    ]);
  });

  test("appendEventsWithCursor advances the cached committed record count", async () => {
    const cursors = await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage;
        const first = yield* storage.sessions.appendEventsWithCursor(sessionId, [
          { type: "a", timestamp: 100 },
          { type: "b", timestamp: 200 },
        ]);
        const second = yield* storage.sessions.appendEventsWithCursor(sessionId, [
          { type: "c", timestamp: 300 },
        ]);
        return { first, second };
      }),
      tmpDir
    );

    expect(cursors).toEqual({
      first: { index: 1, timestamp: 200 },
      second: { index: 2, timestamp: 300 },
    });
  });

  test("deleting a session invalidates its cached event count", async () => {
    const cursor = await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage;
        yield* storage.sessions.appendEventsWithCursor(sessionId, [
          { type: "old", timestamp: 100 },
        ]);
        yield* storage.sessions.deleteSession(sessionId);
        return yield* storage.sessions.appendEventsWithCursor(sessionId, [
          { type: "new", timestamp: 200 },
        ]);
      }),
      tmpDir
    );

    expect(cursor).toEqual({ index: 0, timestamp: 200 });
  });

  test("recovers a torn final event without hiding interior corruption", async () => {
    await mkdir(paths.sessionDir(sessionId), { recursive: true });
    await writeFile(
      paths.sessionEventsFile(sessionId),
      '{"type":"complete"}\n{"type":',
      "utf-8"
    );

    const recovered = await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage;
        return yield* storage.sessions.readEvents<{ type: string }>(sessionId);
      }),
      tmpDir
    );
    expect(recovered).toEqual([{ type: "complete" }]);
    expect(
      await readFile(paths.sessionEventsFile(sessionId), "utf-8")
    ).toBe('{"type":"complete"}\n');

    await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage;
        yield* storage.sessions.appendEventsWithCursor(sessionId, [
          { type: "after-repair", timestamp: 200 },
        ]);
      }),
      tmpDir
    );
    expect(
      await readFile(paths.sessionEventsFile(sessionId), "utf-8")
    ).toBe('{"type":"complete"}\n{"type":"after-repair","timestamp":200}\n');

    await writeFile(
      paths.sessionEventsFile(sessionId),
      '{"type":"complete"}\nnot-json\n{"type":"later"}\n',
      "utf-8"
    );
    const exit = await Effect.runPromise(
      Effect.exit(
        Effect.gen(function* () {
          const storage = yield* MagnitudeStorage;
          return yield* storage.sessions.readEvents<{ type: string }>(
            sessionId
          );
        }).pipe(Effect.provide(testLayer(tmpDir)))
      )
    );
    expect(exit._tag).toBe("Failure");
    expect(
      await readFile(paths.sessionEventsFile(sessionId), "utf-8")
    ).toBe('{"type":"complete"}\nnot-json\n{"type":"later"}\n');
  });

  test.each(["\n", "  \t\n"])(
    "rejects a committed blank JSONL record without changing the file: %j",
    async (blankRecord) => {
      const contents = `{"type":"complete"}\n${blankRecord}{"type":"later"}\n`;
      await mkdir(paths.sessionDir(sessionId), { recursive: true });
      await writeFile(paths.sessionEventsFile(sessionId), contents, "utf-8");

      const exit = await Effect.runPromise(
        Effect.exit(
          Effect.gen(function* () {
            const storage = yield* MagnitudeStorage;
            return yield* storage.sessions.readEvents<{ type: string }>(sessionId);
          }).pipe(Effect.provide(testLayer(tmpDir)))
        )
      );

      expect(exit._tag).toBe("Failure");
      expect(await readFile(paths.sessionEventsFile(sessionId), "utf-8")).toBe(contents);
    }
  );

  test("preserves syntactically valid JSON at the generic storage boundary", async () => {
    const contents = '{"unexpected":true}\n';
    await mkdir(paths.sessionDir(sessionId), { recursive: true });
    await writeFile(paths.sessionEventsFile(sessionId), contents, "utf-8");

    const records = await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage;
        return yield* storage.sessions.readEvents<{ readonly unexpected: boolean }>(sessionId);
      }),
      tmpDir
    );

    expect(records).toEqual([{ unexpected: true }]);
    expect(await readFile(paths.sessionEventsFile(sessionId), "utf-8")).toBe(contents);
  });

  test("preserves a valid final event that is missing only its newline", async () => {
    await mkdir(paths.sessionDir(sessionId), { recursive: true });
    await writeFile(
      paths.sessionEventsFile(sessionId),
      '{"type":"first"}\n{"type":"complete-tail"}',
      "utf-8"
    );

    const recovered = await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage;
        return yield* storage.sessions.readEvents<{ type: string }>(sessionId);
      }),
      tmpDir
    );

    expect(recovered).toEqual([
      { type: "first" },
      { type: "complete-tail" },
    ]);
    expect(
      await readFile(paths.sessionEventsFile(sessionId), "utf-8")
    ).toBe('{"type":"first"}\n{"type":"complete-tail"}\n');
  });

  test("repairs a torn tail before appending without a preceding read", async () => {
    await mkdir(paths.sessionDir(sessionId), { recursive: true });
    await writeFile(
      paths.sessionEventsFile(sessionId),
      '{"type":"first"}\n{"type":',
      "utf-8"
    );

    const cursor = await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage;
        return yield* storage.sessions.appendEventsWithCursor(sessionId, [
          { type: "after-repair", timestamp: 200 },
        ]);
      }),
      tmpDir
    );

    expect(cursor).toEqual({ index: 1, timestamp: 200 });
    expect(
      await readFile(paths.sessionEventsFile(sessionId), "utf-8")
    ).toBe('{"type":"first"}\n{"type":"after-repair","timestamp":200}\n');
  });

  test("repairs a torn UTF-8 tail at a byte boundary", async () => {
    await mkdir(paths.sessionDir(sessionId), { recursive: true });
    await writeFile(
      paths.sessionEventsFile(sessionId),
      '{"type":"complete","text":"héllo"}\n{"type":"torn","text":"世',
      "utf-8"
    );

    const recovered = await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage;
        return yield* storage.sessions.readEvents<{ type: string; text: string }>(sessionId);
      }),
      tmpDir
    );

    expect(recovered).toEqual([{ type: "complete", text: "héllo" }]);
    expect(
      await readFile(paths.sessionEventsFile(sessionId), "utf-8")
    ).toBe('{"type":"complete","text":"héllo"}\n');
  });

  test("projection snapshots are read back as raw json", async () => {
    const snapshot = {
      eventCursor: { index: 0, timestamp: 100 },
      projections: {
        Counter: { count: 1 },
      },
      schemaVersion: "ignored-runtime-version",
    };

    await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage;
        yield* storage.sessions.writeProjectionSnapshot(sessionId, snapshot);
      }),
      tmpDir
    );

    const result = await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage;
        return yield* storage.sessions.readProjectionSnapshot(sessionId);
      }),
      tmpDir
    );

    expect(result).toEqual(snapshot);
  });

  test("addressed entries are stored separately from projection snapshots", async () => {
    const namespace = "DisplayTimeline/messages";
    const address =
      "DisplayTimeline/messages/members/critic%2F2/entries/entry-0";
    const value = {
      items: [
        { id: "m1", text: "hello" },
        { id: "m2", text: null },
      ],
    };

    await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage;
        yield* storage.sessions.writeAddressedEntry(
          sessionId,
          namespace,
          address,
          value
        );
      }),
      tmpDir
    );

    const result = await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage;
        return yield* storage.sessions.readAddressedEntry(
          sessionId,
          namespace,
          address
        );
      }),
      tmpDir
    );

    expect(result).toEqual({ value });
    expect(
      await Bun.file(paths.sessionProjectionSnapshotFile(sessionId)).exists()
    ).toBe(false);
    expect(
      await Bun.file(
        paths.sessionAddressedEntryFile(sessionId, namespace, address)
      ).exists()
    ).toBe(true);
  });

  test("addressed entries preserve null values distinctly from missing entries", async () => {
    await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage;
        yield* storage.sessions.writeAddressedEntry(
          sessionId,
          "ns",
          "addr",
          null
        );
      }),
      tmpDir
    );

    const result = await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage;
        const present = yield* storage.sessions.readAddressedEntry(
          sessionId,
          "ns",
          "addr"
        );
        const missing = yield* storage.sessions.readAddressedEntry(
          sessionId,
          "ns",
          "missing"
        );
        return { present, missing };
      }),
      tmpDir
    );

    expect(result.present).toEqual({ value: null });
    expect(result.missing).toBeNull();
  });

  test("addressed entries expose stored byte size without reading payloads", async () => {
    const namespace = "DisplayTimeline/messages";
    const address = "DisplayTimeline/messages/entries/entry-0";

    const result = await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage;
        yield* storage.sessions.writeAddressedEntry(
          sessionId,
          namespace,
          address,
          {
            items: [{ id: "m1", text: "hello" }],
          }
        );

        return {
          present: yield* storage.sessions.statAddressedEntry(
            sessionId,
            namespace,
            address
          ),
          missing: yield* storage.sessions.statAddressedEntry(
            sessionId,
            namespace,
            "missing"
          ),
        };
      }),
      tmpDir
    );

    expect(result.present?.storedBytes).toBeGreaterThan(0);
    expect(result.missing).toBeNull();
  });

  test("addressed entry namespace and address path parts are encoded independently", async () => {
    const left = {
      namespace: "a/b",
      address: "c",
      value: { side: "left" },
    };
    const right = {
      namespace: "a",
      address: "b/c",
      value: { side: "right" },
    };

    await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage;
        yield* storage.sessions.writeAddressedEntry(
          sessionId,
          left.namespace,
          left.address,
          left.value
        );
        yield* storage.sessions.writeAddressedEntry(
          sessionId,
          right.namespace,
          right.address,
          right.value
        );
      }),
      tmpDir
    );

    const result = await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage;
        return {
          left: yield* storage.sessions.readAddressedEntry(
            sessionId,
            left.namespace,
            left.address
          ),
          right: yield* storage.sessions.readAddressedEntry(
            sessionId,
            right.namespace,
            right.address
          ),
        };
      }),
      tmpDir
    );

    expect(result.left).toEqual({ value: left.value });
    expect(result.right).toEqual({ value: right.value });
    expect(
      paths.sessionAddressedEntryFile(sessionId, left.namespace, left.address)
    ).not.toBe(
      paths.sessionAddressedEntryFile(sessionId, right.namespace, right.address)
    );
  });

  test("deleteSession removes non-empty session directory", async () => {
    await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage;
        yield* storage.sessions.writeMeta(sessionId, baseMeta as any);
      }),
      tmpDir
    );
    await mkdir(join(paths.sessionDir(sessionId), "scratchpad", "nested"), {
      recursive: true,
    });
    await writeFile(
      join(paths.sessionDir(sessionId), "scratchpad", "nested", "note.txt"),
      "hello",
      "utf-8"
    );

    await run(
      Effect.gen(function* () {
        const storage = yield* MagnitudeStorage;
        yield* storage.sessions.deleteSession(sessionId);
      }),
      tmpDir
    );

    expect(await Bun.file(paths.sessionDir(sessionId)).exists()).toBe(false);
  });
});
