import { afterEach, describe, expect, it } from "vitest";
import { FetchHttpClient } from "@effect/platform";
import { BunContext } from "@effect/platform-bun";
import { Effect } from "effect";
import { chmod, mkdir, mkdtemp, stat, utimes, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { tmpdir } from "node:os";
import { makeLocalDaemonSpawner } from "./local-daemon-spawner";

describe("local daemon spawner rendezvous", () => {
  let server: ReturnType<typeof Bun.serve> | undefined;

  afterEach(() => {
    server?.stop(true);
    server = undefined;
  });

  it.runIf(process.platform !== "win32")(
    "threads the elected data root into the resolved ACN command",
    async () => {
      const dataDir = await mkdtemp(join(tmpdir(), "magnitude-spawner-data-root-"));
      const version = "1.2.3";
      const id = "data-root-owner";
      const pid = 6543;
      const binary = join(dataDir, "fake-acn");
      await writeFile(binary, `#!/bin/sh\nprintf '%s\\n' '${version}'\n`);
      await chmod(binary, 0o755);
      server = Bun.serve({
        port: 0,
        fetch: () => Response.json({ service: "magnitude-acn", version, id, pid }),
      });
      const registryDirectory = join(dataDir, "acn");
      const registryPath = join(registryDirectory, "registry.json");
      await mkdir(registryDirectory, { recursive: true });
      let launched: readonly string[] | null = null;

      const spawnProcess = (command: readonly string[]) => {
        launched = command;
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

      const url = await makeLocalDaemonSpawner(spawnProcess, {
        binaryPath: binary,
        dataDir,
        version,
        spawnTimeoutMs: 2_000,
        probeTimeoutMs: 200,
      }).pipe(
        Effect.flatMap((spawner) => spawner.spawn(undefined)),
        Effect.provide([BunContext.layer, FetchHttpClient.layer]),
        Effect.runPromise,
      );

      expect(url).toBe(`http://127.0.0.1:${server.port}`);
      expect(launched).toEqual([binary, "serve", "--register", "--data-dir", dataDir]);
    },
  );

  it("accepts a newer winner after its older candidate exits and releases election", async () => {
    const dataDir = await mkdtemp(join(tmpdir(), "magnitude-spawner-"));
    const candidateVersion = "1.0.0";
    const winnerVersion = "2.0.0";
    const id = "winning-owner";
    const pid = 4321;
    server = Bun.serve({
      port: 0,
      fetch: () =>
        Response.json({
          service: "magnitude-acn",
          version: winnerVersion,
          id,
          pid,
        }),
    });
    const registryDirectory = join(dataDir, "acn");
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
              version: winnerVersion,
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
      version: candidateVersion,
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
    const registryDirectory = join(dataDir, "acn");
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
    const electionDirectory = join(dataDir, "acn", "spawn-election");
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
    const electionDirectory = join(dataDir, "acn", "spawn-election");
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

  it("waits for an incompatible owner process to exit before spawning its replacement", async () => {
    const dataDir = await mkdtemp(join(tmpdir(), "magnitude-spawner-upgrade-"));
    const registryDirectory = join(dataDir, "acn");
    const registryPath = join(registryDirectory, "registry.json");
    await mkdir(registryDirectory, { recursive: true });

    const old = Bun.spawn({
      cmd: [
        process.execPath,
        "-e",
        "process.on('SIGTERM',()=>setTimeout(()=>process.exit(0),50));setInterval(()=>{},1000)",
      ],
      stdout: "ignore",
      stderr: "ignore",
    });
    let health = { version: "1.0.0", id: "old-owner", pid: old.pid };
    let authenticatedShutdown = false;
    server = Bun.serve({
      port: 0,
      fetch: (request) => {
        if (request.method === "POST") {
          authenticatedShutdown = request.headers.get("authorization") === "Bearer takeover-token";
          if (authenticatedShutdown) process.kill(old.pid, "SIGTERM");
          return new Response(null, { status: authenticatedShutdown ? 202 : 401 });
        }
        if (authenticatedShutdown && health.version === "1.0.0") {
          return new Response(null, { status: 503 });
        }
        return Response.json({ service: "magnitude-acn", ...health });
      },
    });
    await writeFile(
      registryPath,
      JSON.stringify({
        schemaVersion: 1,
        registration: {
          ...health,
          url: `http://127.0.0.1:${server.port}`,
          timestamp: Date.now(),
          shutdownToken: "takeover-token",
        },
      }),
    );

    let oldWasGoneAtSpawn = false;
    const spawnProcess = () => {
      try {
        process.kill(old.pid, 0);
      } catch {
        oldWasGoneAtSpawn = true;
      }
      health = { version: "2.0.0", id: "new-owner", pid: 9876 };
      void writeFile(
        registryPath,
        JSON.stringify({
          schemaVersion: 1,
          registration: {
            ...health,
            url: `http://127.0.0.1:${server!.port}`,
            timestamp: Date.now(),
          },
        }),
      );
      return { pid: health.pid, exited: new Promise<number | null>(() => {}) };
    };

    const url = await makeLocalDaemonSpawner(spawnProcess, {
      dataDir,
      version: "2.0.0",
      spawnTimeoutMs: 2000,
      probeTimeoutMs: 200,
    }).pipe(
      Effect.flatMap((spawner) => spawner.spawn(["ignored"])),
      Effect.provide([BunContext.layer, FetchHttpClient.layer]),
      Effect.runPromise,
    );

    expect(url).toBe(`http://127.0.0.1:${server.port}`);
    expect(oldWasGoneAtSpawn).toBe(true);
    expect(authenticatedShutdown).toBe(true);
    expect([0, 143]).toContain(await old.exited);
  });

  it("lets an older client reuse a newer healthy owner without spawning", async () => {
    const dataDir = await mkdtemp(join(tmpdir(), "magnitude-spawner-downgrade-"));
    const registryDirectory = join(dataDir, "acn");
    const registryPath = join(registryDirectory, "registry.json");
    await mkdir(registryDirectory, { recursive: true });
    const incumbent = { version: "2.0.0", id: "newer-owner", pid: process.pid };
    server = Bun.serve({
      port: 0,
      fetch: () => Response.json({ service: "magnitude-acn", ...incumbent }),
    });
    await writeFile(registryPath, JSON.stringify({
      schemaVersion: 1,
      registration: {
        ...incumbent,
        url: `http://127.0.0.1:${server.port}`,
        timestamp: Date.now(),
      },
    }));
    let spawned = false;

    const url = await makeLocalDaemonSpawner(
      () => {
        spawned = true;
        return { pid: 9999, exited: Promise.resolve(1) };
      },
      { dataDir, version: "1.0.0", spawnTimeoutMs: 500, probeTimeoutMs: 100 },
    ).pipe(
      Effect.flatMap((spawner) => spawner.spawn(["ignored"])),
      Effect.provide([BunContext.layer, FetchHttpClient.layer]),
      Effect.runPromise,
    );

    expect(url).toBe(`http://127.0.0.1:${server.port}`);
    expect(spawned).toBe(false);
  });
});
