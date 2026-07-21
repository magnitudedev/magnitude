import { afterEach, describe, expect, it } from "vitest";
import { FetchHttpClient } from "@effect/platform";
import { BunContext } from "@effect/platform-bun";
import { Effect } from "effect";
import { mkdir, mkdtemp, stat, utimes, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { tmpdir } from "node:os";
import { makeLocalDaemonSpawner } from "./local-daemon-spawner";

describe("local daemon spawner rendezvous", () => {
  let server: ReturnType<typeof Bun.serve> | undefined;

  afterEach(() => {
    server?.stop(true);
    server = undefined;
  });

  it("accepts another candidate after its own candidate exits and releases election", async () => {
    const dataDir = await mkdtemp(join(tmpdir(), "magnitude-spawner-"));
    const version = "test-version";
    const id = "winning-owner";
    const pid = 4321;
    server = Bun.serve({
      port: 0,
      fetch: () =>
        Response.json({
          service: "magnitude-acn",
          version,
          id,
          pid,
        }),
    });
    const registryDirectory = join(dataDir, "acn", encodeURIComponent(version));
    const registryPath = join(registryDirectory, "registry.json");
    await mkdir(registryDirectory, { recursive: true });

    const spawnProcess = () => {
      setTimeout(() => {
        void writeFile(
          registryPath,
          JSON.stringify({
            schemaVersion: 1,
            registration: {
              id,
              version,
              url: `http://127.0.0.1:${server!.port}`,
              pid,
              timestamp: Date.now(),
            },
          })
        );
      }, 25);
      return { pid: 9999, exited: Promise.resolve(1) };
    };

    const url = await makeLocalDaemonSpawner(spawnProcess, {
      dataDir,
      version,
      spawnTimeoutMs: 2000,
      probeTimeoutMs: 200,
    }).pipe(
      Effect.flatMap((spawner) => spawner.spawn(["ignored"])),
      Effect.provide([BunContext.layer, FetchHttpClient.layer]),
      Effect.runPromise
    );

    expect(url).toBe(`http://127.0.0.1:${server.port}`);
    expect(
      await Bun.file(
        join(registryDirectory, "spawn-election")
      ).exists()
    ).toBe(false);
  });

  it("single-flights spawn across independent local spawners", async () => {
    const dataDir = await mkdtemp(join(tmpdir(), "magnitude-spawner-election-"));
    const version = "test-election-version";
    const id = "election-winner";
    const pid = 7654;
    let spawnCalls = 0;
    server = Bun.serve({
      port: 0,
      fetch: () => Response.json({ service: "magnitude-acn", version, id, pid }),
    });
    const registryDirectory = join(dataDir, "acn", encodeURIComponent(version));
    const registryPath = join(registryDirectory, "registry.json");
    await mkdir(registryDirectory, { recursive: true });

    const spawnProcess = () => {
      spawnCalls++;
      setTimeout(() => {
        void writeFile(registryPath, JSON.stringify({
          schemaVersion: 1,
          registration: {
            id,
            version,
            url: `http://127.0.0.1:${server!.port}`,
            pid,
            timestamp: Date.now(),
          },
        }));
      }, 25);
      return { pid: 9999, exited: new Promise<number | null>(() => {}) };
    };

    const makeSpawner = () => makeLocalDaemonSpawner(spawnProcess, {
      dataDir,
      version,
      spawnTimeoutMs: 2000,
      probeTimeoutMs: 200,
    }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer]));
    const [first, second] = await Effect.all([makeSpawner(), makeSpawner()]).pipe(
      Effect.flatMap(([left, right]) => Effect.all([
        left.spawn(["ignored"]),
        right.spawn(["ignored"]),
      ], { concurrency: "unbounded" })),
      Effect.runPromise,
    );

    expect(first).toBe(`http://127.0.0.1:${server.port}`);
    expect(second).toBe(first);
    expect(spawnCalls).toBe(1);
    expect(await Bun.file(join(registryDirectory, "spawn-election")).exists()).toBe(false);
  });

  it("does not release another process's election when a contender times out", async () => {
    const dataDir = await mkdtemp(join(tmpdir(), "magnitude-spawner-foreign-election-"));
    const version = "test-foreign-election-version";
    const electionDirectory = join(dataDir, "acn", encodeURIComponent(version), "spawn-election");
    await mkdir(dirname(electionDirectory), { recursive: true });
    await writeFile(electionDirectory, JSON.stringify({
      token: "foreign-owner",
      pid: process.pid,
    }));
    const staleTimestamp = new Date(Date.now() - 120_000);
    await utimes(electionDirectory, staleTimestamp, staleTimestamp);

    const attempt = makeLocalDaemonSpawner(
      () => ({ pid: 9999, exited: new Promise<number | null>(() => {}) }),
      {
        dataDir,
        version,
        spawnTimeoutMs: 100,
        probeTimeoutMs: 20,
      },
    ).pipe(
      Effect.flatMap((spawner) => spawner.spawn(["ignored"])),
      Effect.provide([BunContext.layer, FetchHttpClient.layer]),
      Effect.runPromise,
    );

    await expect(attempt).rejects.toThrow("Timed out waiting for ACN spawn election");
    expect(JSON.parse(await Bun.file(electionDirectory).text())).toEqual({
      token: "foreign-owner",
      pid: process.pid,
    });
  });

  it("recovers a dead election within the current spawn wait budget", async () => {
    const dataDir = await mkdtemp(join(tmpdir(), "magnitude-spawner-dead-election-"));
    const version = "test-dead-election-version";
    const electionDirectory = join(dataDir, "acn", encodeURIComponent(version), "spawn-election");
    await mkdir(dirname(electionDirectory), { recursive: true });
    await writeFile(electionDirectory, JSON.stringify({
      token: "dead-owner",
      pid: 2_147_483_647,
    }));
    const staleTimestamp = new Date(Date.now() - 1_000);
    await utimes(electionDirectory, staleTimestamp, staleTimestamp);

    let spawnCalls = 0;
    const attempt = makeLocalDaemonSpawner(
      () => {
        spawnCalls++;
        return { pid: 9999, exited: Promise.resolve(1) };
      },
      {
        dataDir,
        version,
        spawnTimeoutMs: 100,
        probeTimeoutMs: 20,
      },
    ).pipe(
      Effect.flatMap((spawner) => spawner.spawn(["ignored"])),
      Effect.provide([BunContext.layer, FetchHttpClient.layer]),
      Effect.runPromise,
    );

    await expect(attempt).rejects.toThrow("No compatible ACN became healthy");
    expect(spawnCalls).toBe(1);
    expect(await Bun.file(electionDirectory).exists()).toBe(false);
    expect((await stat(`${electionDirectory}.stale-dead-owner`)).isFile()).toBe(true);
  });
});
