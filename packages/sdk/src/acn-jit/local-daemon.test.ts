import { afterEach, describe, expect, it } from "vitest";
import { FetchHttpClient } from "@effect/platform";
import { BunContext } from "@effect/platform-bun";
import { Effect, Option, Stream } from "effect";
import {
  chmod,
  mkdir,
  mkdtemp,
  stat,
  utimes,
  writeFile,
} from "node:fs/promises";
import { dirname, join } from "node:path";
import { tmpdir } from "node:os";
import {
  ChildProcessSpawner,
  makeLocalDaemonDiscovery,
  makeLocalDaemonLauncher,
  type ChildProcess,
} from "./local-daemon";
import { runDaemonLaunch, type DaemonLauncher } from "./daemon-launcher";

const spawn = (
  spawner: DaemonLauncher,
  command: Option.Option<ReadonlyArray<string>>
) => runDaemonLaunch(spawner.launch(command));

const testProcess = (
  exited: Promise<number | null> = new Promise(() => {}),
  pid = 9999
): ChildProcess => ({
  pid: Option.some(pid),
  exited: Effect.promise(() => exited).pipe(Effect.map((code) => code ?? 1)),
  diagnostic: Effect.succeed(Option.none()),
  kill: () => Effect.void,
});

const testSpawner = (
  spawnProcess: (command: ReadonlyArray<string>) => {
    readonly pid: number;
    readonly exited: Promise<number | null>;
  }
): ChildProcessSpawner => ({
  spawn: (command) =>
    Effect.sync(() => {
      const process = spawnProcess(command);
      return testProcess(process.exited, process.pid);
    }),
});

describe("local daemon spawner rendezvous", () => {
  let server: ReturnType<typeof Bun.serve> | undefined;

  afterEach(() => {
    server?.stop(true);
    server = undefined;
  });

  it.runIf(process.platform !== "win32")(
    "threads the elected data root into the resolved ACN command",
    async () => {
      const dataDir = await mkdtemp(
        join(tmpdir(), "magnitude-spawner-data-root-")
      );
      const version = "1.2.3";
      const id = "data-root-owner";
      const pid = 6543;
      const binary = join(dataDir, "fake-acn");
      await writeFile(binary, `#!/bin/sh\nprintf '%s\\n' '${version}'\n`);
      await chmod(binary, 0o755);
      server = Bun.serve({
        port: 0,
        fetch: () =>
          Response.json({
            service: "magnitude-acn",
            version,
            id,
            pid,
            state: { _tag: "Ready" },
          }),
      });
      const registryDirectory = join(dataDir, "acn");
      const registryPath = join(registryDirectory, "registry.json");
      await mkdir(registryDirectory, { recursive: true });
      let launched: readonly string[] | null = null;

      const spawnProcess = (command: readonly string[]) => {
        launched = command;
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
        return { pid: 9999, exited: new Promise<number | null>(() => {}) };
      };

      const url = await makeLocalDaemonLauncher({
        binaryPath: binary,
        dataDir,
        version,
        publicationTimeoutMs: 2_000,
        probeTimeoutMs: 200,
      }).pipe(
        Effect.provideService(ChildProcessSpawner, testSpawner(spawnProcess)),
        Effect.flatMap((spawner) => spawn(spawner, Option.none())),
        Effect.provide([BunContext.layer, FetchHttpClient.layer]),
        Effect.runPromise
      );

      expect(url.url).toBe(`http://127.0.0.1:${server.port}`);
      expect(launched).toEqual([
        binary,
        "serve",
        "--register",
        "--data-dir",
        dataDir,
      ]);
    }
  );

  it("fails immediately when its exact spawned candidate exits", async () => {
    const dataDir = await mkdtemp(join(tmpdir(), "magnitude-spawner-"));
    const candidateVersion = "1.0.0";
    const registryDirectory = join(dataDir, "acn");
    await mkdir(registryDirectory, { recursive: true });

    const spawnProcess = () => ({
      pid: 9999,
      exited: Promise.resolve(1),
    });

    const attempt = makeLocalDaemonLauncher({
      dataDir,
      version: candidateVersion,
      publicationTimeoutMs: 2000,
      probeTimeoutMs: 200,
    }).pipe(
      Effect.provideService(ChildProcessSpawner, testSpawner(spawnProcess)),
      Effect.flatMap((spawner) => spawn(spawner, Option.some(["ignored"]))),
      Effect.provide([BunContext.layer, FetchHttpClient.layer]),
      Effect.runPromise
    );

    await expect(attempt).rejects.toThrow('"exitCode": 1');
    expect(
      await Bun.file(join(registryDirectory, "spawn-election")).exists()
    ).toBe(false);
  });

  it("includes the bounded candidate diagnostic in an exact-exit failure", async () => {
    const dataDir = await mkdtemp(
      join(tmpdir(), "magnitude-spawner-diagnostic-")
    );
    const processSpawner: ChildProcessSpawner = {
      spawn: () =>
        Effect.succeed({
          pid: Option.some(9999),
          exited: Effect.succeed(17),
          diagnostic: Effect.succeed(
            Option.some("local inference contract rejected")
          ),
          kill: () => Effect.void,
        }),
    };

    const attempt = makeLocalDaemonLauncher({
      dataDir,
      version: "1.0.0",
      publicationTimeoutMs: 2_000,
      probeTimeoutMs: 200,
    }).pipe(
      Effect.provideService(ChildProcessSpawner, processSpawner),
      Effect.flatMap((spawner) => spawn(spawner, Option.some(["ignored"]))),
      Effect.provide([BunContext.layer, FetchHttpClient.layer]),
      Effect.runPromise
    );

    await expect(attempt).rejects.toThrow("local inference contract rejected");
  });

  it("single-flights spawn across independent local spawners", async () => {
    const dataDir = await mkdtemp(
      join(tmpdir(), "magnitude-spawner-election-")
    );
    const version = "test-election-version";
    const id = "election-winner";
    const pid = 7654;
    let spawnCalls = 0;
    server = Bun.serve({
      port: 0,
      fetch: () =>
        Response.json({
          service: "magnitude-acn",
          version,
          id,
          pid,
          state: { _tag: "Ready" },
        }),
    });
    const registryDirectory = join(dataDir, "acn");
    const registryPath = join(registryDirectory, "registry.json");
    await mkdir(registryDirectory, { recursive: true });

    const spawnProcess = () => {
      spawnCalls++;
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
      return { pid: 9999, exited: new Promise<number | null>(() => {}) };
    };

    const makeSpawner = () =>
      makeLocalDaemonLauncher({
        dataDir,
        version,
        publicationTimeoutMs: 2000,
        probeTimeoutMs: 200,
      }).pipe(
        Effect.provideService(ChildProcessSpawner, testSpawner(spawnProcess)),
        Effect.provide([BunContext.layer, FetchHttpClient.layer])
      );
    const [first, second] = await Effect.all([
      makeSpawner(),
      makeSpawner(),
    ]).pipe(
      Effect.flatMap(([left, right]) =>
        Effect.all(
          [
            spawn(left, Option.some(["ignored"])),
            spawn(right, Option.some(["ignored"])),
          ],
          { concurrency: "unbounded" }
        )
      ),
      Effect.runPromise
    );

    expect(first.url).toBe(`http://127.0.0.1:${server.port}`);
    expect(second).toEqual(first);
    expect(spawnCalls).toBe(1);
    expect(
      await Bun.file(join(registryDirectory, "spawn-election")).exists()
    ).toBe(false);
  });

  it("does not let an expired election block startup even when its PID is live", async () => {
    const dataDir = await mkdtemp(
      join(tmpdir(), "magnitude-spawner-foreign-election-")
    );
    const version = "test-foreign-election-version";
    const electionDirectory = join(dataDir, "acn", "spawn-election");
    await mkdir(dirname(electionDirectory), { recursive: true });
    await writeFile(
      electionDirectory,
      JSON.stringify({
        token: "foreign-owner",
        pid: process.pid,
      })
    );
    const staleTimestamp = new Date(Date.now() - 120_000);
    await utimes(electionDirectory, staleTimestamp, staleTimestamp);

    const spawnProcess = () => ({
      pid: 9999,
      exited: new Promise<number | null>(() => {}),
    });
    const attempt = makeLocalDaemonLauncher({
      dataDir,
      version,
      publicationTimeoutMs: 100,
      probeTimeoutMs: 20,
    }).pipe(
      Effect.provideService(ChildProcessSpawner, testSpawner(spawnProcess)),
      Effect.flatMap((processes) => spawn(processes, Option.some(["ignored"]))),
      Effect.provide([BunContext.layer, FetchHttpClient.layer]),
      Effect.timeoutOption("100 millis"),
      Effect.runPromise
    );

    await expect(attempt).resolves.toEqual(Option.none());
    expect(await Bun.file(electionDirectory).exists()).toBe(false);
  });

  it("recovers a dead election and observes exact candidate exit", async () => {
    const dataDir = await mkdtemp(
      join(tmpdir(), "magnitude-spawner-dead-election-")
    );
    const version = "test-dead-election-version";
    const electionDirectory = join(dataDir, "acn", "spawn-election");
    await mkdir(dirname(electionDirectory), { recursive: true });
    await writeFile(
      electionDirectory,
      JSON.stringify({
        token: "dead-owner",
        pid: 2_147_483_647,
      })
    );
    const staleTimestamp = new Date(Date.now() - 120_000);
    await utimes(electionDirectory, staleTimestamp, staleTimestamp);

    let spawnCalls = 0;
    const spawnProcess = () => {
      spawnCalls++;
      return { pid: 9999, exited: Promise.resolve(1) };
    };
    const attempt = makeLocalDaemonLauncher({
      dataDir,
      version,
      publicationTimeoutMs: 100,
      probeTimeoutMs: 20,
    }).pipe(
      Effect.provideService(ChildProcessSpawner, testSpawner(spawnProcess)),
      Effect.flatMap((spawner) => spawn(spawner, Option.some(["ignored"]))),
      Effect.provide([BunContext.layer, FetchHttpClient.layer]),
      Effect.runPromise
    );

    await expect(attempt).rejects.toThrow('"exitCode": 1');
    expect(spawnCalls).toBe(1);
    expect(await Bun.file(electionDirectory).exists()).toBe(false);
    expect((await stat(`${electionDirectory}.stale-dead-owner`)).isFile()).toBe(
      true
    );
  });

  it("starts a same-base development successor without directing the published owner to shut down", async () => {
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
    const publishedVersion = "1.0.0-alpha.1";
    const developmentVersion = "1.0.0-alpha.1+dev.commit.1";
    let health = { version: publishedVersion, id: "old-owner", pid: old.pid };
    let shutdownRequests = 0;
    server = Bun.serve({
      port: 0,
      fetch: (request) => {
        if (request.method === "POST") {
          shutdownRequests++;
          return new Response(null, { status: 405 });
        }
        return Response.json({
          service: "magnitude-acn",
          ...health,
          ...(health.version === publishedVersion
            ? {}
            : { state: { _tag: "Ready" } }),
        });
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
        },
      })
    );

    let oldWasGoneAtSpawn = false;
    const spawnProcess = () => {
      try {
        process.kill(old.pid, 0);
      } catch {
        oldWasGoneAtSpawn = true;
      }
      health = { version: developmentVersion, id: "new-owner", pid: 9876 };
      void writeFile(
        registryPath,
        JSON.stringify({
          schemaVersion: 1,
          registration: {
            ...health,
            url: `http://127.0.0.1:${server!.port}`,
            timestamp: Date.now(),
          },
        })
      );
      return { pid: health.pid, exited: new Promise<number | null>(() => {}) };
    };

    const url = await makeLocalDaemonLauncher({
      dataDir,
      version: developmentVersion,
      publicationTimeoutMs: 2000,
      probeTimeoutMs: 200,
    }).pipe(
      Effect.provideService(ChildProcessSpawner, testSpawner(spawnProcess)),
      Effect.flatMap((spawner) => spawn(spawner, Option.some(["ignored"]))),
      Effect.provide([BunContext.layer, FetchHttpClient.layer]),
      Effect.runPromise
    );

    expect(url.url).toBe(`http://127.0.0.1:${server.port}`);
    expect(oldWasGoneAtSpawn).toBe(false);
    expect(shutdownRequests).toBe(0);
    process.kill(old.pid, "SIGTERM");
    expect([0, 143]).toContain(await old.exited);
  });

  it("reports a newer healthy canonical daemon without applying client policy", async () => {
    const dataDir = await mkdtemp(
      join(tmpdir(), "magnitude-spawner-downgrade-")
    );
    const registryDirectory = join(dataDir, "acn");
    const registryPath = join(registryDirectory, "registry.json");
    await mkdir(registryDirectory, { recursive: true });
    const incumbent = { version: "2.0.0", id: "newer-owner", pid: process.pid };
    server = Bun.serve({
      port: 0,
      fetch: () =>
        Response.json({
          service: "magnitude-acn",
          ...incumbent,
          state: { _tag: "Ready" },
        }),
    });
    await writeFile(
      registryPath,
      JSON.stringify({
        schemaVersion: 1,
        registration: {
          ...incumbent,
          url: `http://127.0.0.1:${server.port}`,
          timestamp: Date.now(),
        },
      })
    );
    const url = await makeLocalDaemonDiscovery({
      dataDir,
      probeTimeoutMs: 100,
    }).pipe(
      Effect.flatMap((processes) => processes.current()),
      Effect.provide([BunContext.layer, FetchHttpClient.layer]),
      Effect.runPromise
    );

    expect(Option.getOrThrow(url).url).toBe(`http://127.0.0.1:${server.port}`);
  });
});
