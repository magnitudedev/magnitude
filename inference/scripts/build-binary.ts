import {
  copyFile,
  mkdir,
  mkdtemp,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { $ } from "bun";
import {
  getDefaultBunTarget,
  getTargetInfo,
} from "../../scripts/release-target";

const PROJECT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..");

export type IcnReleaseBackend = "cpu" | "cuda";

export const icnReleasePlatformKey = (
  platformKey: string,
  backend: IcnReleaseBackend
): string => backend === "cuda" ? `${platformKey}-cuda` : platformKey;

function rustTarget(target: string): string {
  const { platform, arch } = getTargetInfo(target);
  const mapped: Record<string, string> = {
    "darwin-arm64": "aarch64-apple-darwin",
    "darwin-x64": "x86_64-apple-darwin",
    "linux-arm64": "aarch64-unknown-linux-gnu",
    "linux-x64": "x86_64-unknown-linux-gnu",
    "windows-x64": "x86_64-pc-windows-msvc",
  };
  const value = mapped[`${platform}-${arch}`];
  if (!value) throw new Error(`No ICN Rust target for ${target}`);
  return value;
}

async function fileSha256(path: string): Promise<string> {
  const bytes = await Bun.file(path).arrayBuffer();
  return Buffer.from(await crypto.subtle.digest("SHA-256", bytes)).toString(
    "hex"
  );
}

interface IcnBuildIdentity {
  readonly api_version: number;
  readonly native_build: string;
  readonly target: string;
  readonly backends: readonly string[];
}

async function readIcnIdentity(binaryFile: string): Promise<IcnBuildIdentity> {
  const version = Bun.spawn([binaryFile, "version", "--json"], {
    stdout: "pipe",
    stderr: "pipe",
  });
  const stdout = await new Response(version.stdout).text();
  const stderr = await new Response(version.stderr).text();
  if ((await version.exited) !== 0) {
    throw new Error(`[build:icn] version smoke failed: ${stderr.trim()}`);
  }
  const identity = JSON.parse(stdout) as Partial<IcnBuildIdentity>;
  if (
    identity.api_version !== 1
    || typeof identity.native_build !== "string"
    || !identity.native_build
    || typeof identity.target !== "string"
    || !identity.target
    || !Array.isArray(identity.backends)
    || !identity.backends.every((backend) => typeof backend === "string")
  ) {
    throw new Error("[build:icn] version smoke returned an invalid identity");
  }
  return identity as IcnBuildIdentity;
}

async function smokeIcn(binaryFile: string): Promise<void> {
  await readIcnIdentity(binaryFile);

  const doctor = Bun.spawn([binaryFile, "doctor"], {
    stdout: "pipe",
    stderr: "pipe",
  });
  const doctorOutput = await new Response(doctor.stdout).text();
  if ((await doctor.exited) !== 0 || !doctorOutput.includes("successfully")) {
    throw new Error("[build:icn] doctor smoke failed");
  }

  const modelStore = await mkdtemp(join(tmpdir(), "magnitude-icn-smoke-"));
  const cacheRoot = await mkdtemp(join(tmpdir(), "magnitude-icn-cache-smoke-"));
  const capability = crypto.randomUUID();
  const child = Bun.spawn(
    [
      binaryFile,
      "serve",
      "--bind",
      "127.0.0.1:0",
      "--instance-id",
      "release-smoke",
      "--parent-pid",
      String(process.pid),
      "--model-store",
      modelStore,
      "--cache-root",
      cacheRoot,
    ],
    {
      stdout: "pipe",
      stderr: "pipe",
      env: { ...process.env, MAGNITUDE_ICN_AUTH_TOKEN: capability },
    }
  );
  try {
    const reader = child.stdout.getReader();
    const decoder = new TextDecoder();
    let output = "";
    let startupTimeout: ReturnType<typeof setTimeout> | undefined;
    const startup = await Promise.race([
      (async () => {
        while (output.length <= 64 * 1024) {
          const next = await reader.read();
          if (next.done) break;
          output += decoder.decode(next.value, { stream: true });
          const line = output
            .split(/\r?\n/)
            .find((value) => value.startsWith("MAGNITUDE_ICN_READY "));
          if (line)
            return JSON.parse(line.slice("MAGNITUDE_ICN_READY ".length));
        }
        throw new Error("ICN exited without a bounded startup record");
      })(),
      new Promise<never>((_, reject) => {
        startupTimeout = setTimeout(
          () => reject(new Error("ICN startup smoke timed out")),
          30_000
        );
      }),
    ]).finally(() => clearTimeout(startupTimeout));
    const health = await fetch(`${startup.origin}/health`).then((response) =>
      response.json()
    );
    if (!health.ready || health.instanceId !== "release-smoke") {
      throw new Error("ICN health smoke returned the wrong identity");
    }
    const runtime = await fetch(`${startup.origin}/v1/runtime`, {
      headers: { authorization: `Bearer ${capability}` },
    });
    if (!runtime.ok) throw new Error("ICN authenticated runtime smoke failed");
  } finally {
    child.kill("SIGTERM");
    await child.exited;
    await rm(modelStore, { recursive: true, force: true });
  }
}

export async function buildIcnBinary(
  target: string,
  outDir = resolve(PROJECT_ROOT, "release"),
  backend: IcnReleaseBackend = "cpu"
): Promise<string> {
  const info = getTargetInfo(target);
  if (backend === "cuda" && info.platform !== "linux") {
    throw new Error("[build:icn] CUDA release artifacts require a Linux target");
  }
  const cargoTarget = rustTarget(target);
  const releasePlatformKey = icnReleasePlatformKey(info.platformKey, backend);
  const binDir = resolve(PROJECT_ROOT, "bin");
  const cargoTargetDir = resolve(
    PROJECT_ROOT,
    "inference/target",
    `release-${backend}`
  );
  const manifestPath = resolve(PROJECT_ROOT, "inference/Cargo.toml");
  const name = "magnitude-icn" + info.executableExt;
  const companionManifestName = "magnitude-icn-manifest.json";
  const tarballPath = resolve(
    outDir,
    `magnitude-icn-${releasePlatformKey}.tar.gz`
  );

  await mkdir(binDir, { recursive: true });
  await mkdir(outDir, { recursive: true });
  console.log(`[build:icn] Building ${cargoTarget} (${backend})`);
  const cargo = [
    "cargo",
    "build",
    "--release",
    "--manifest-path",
    manifestPath,
    "-p",
    "icn-server",
    "--target",
    cargoTarget,
    ...(backend === "cuda" ? ["--features", "cuda-no-vmm"] : []),
  ];
  const build = Bun.spawn(cargo, {
    cwd: PROJECT_ROOT,
    env: { ...process.env, CARGO_TARGET_DIR: cargoTargetDir },
    stdin: "inherit",
    stdout: "inherit",
    stderr: "inherit",
  });
  if ((await build.exited) !== 0) {
    throw new Error(`[build:icn] ${backend} release build failed`);
  }

  const source = resolve(
    cargoTargetDir,
    cargoTarget,
    "release",
    name
  );
  const destination = resolve(binDir, name);
  await copyFile(source, destination);
  if (info.platform === "darwin") {
    await $`codesign --force --deep --sign - ${destination}`;
  }
  const identity = await readIcnIdentity(destination);
  if (backend === "cuda" && !identity.backends.includes("cuda")) {
    throw new Error("[build:icn] CUDA artifact identity does not include CUDA");
  }
  if (backend === "cpu" && identity.backends.includes("cuda")) {
    throw new Error("[build:icn] CPU artifact identity unexpectedly includes CUDA");
  }
  if (target === getDefaultBunTarget() && backend === "cpu") {
    await smokeIcn(destination);
  }
  await writeFile(
    resolve(binDir, companionManifestName),
    JSON.stringify(
      {
        schemaVersion: 1,
        binary: name,
        sha256: await fileSha256(destination),
        apiVersion: identity.api_version,
        nativeBuild: identity.native_build,
        target: identity.target,
        backends: identity.backends,
      },
      null,
      2
    ) + "\n"
  );
  await $`tar -czf ${tarballPath} -C ${binDir} ${name} ${companionManifestName}`;
  console.log("[build:icn] Created " + tarballPath);
  return tarballPath;
}

if (import.meta.main) {
  const targetArg = process.argv[2] ?? getDefaultBunTarget();
  const outDirArg = process.argv[3] ? resolve(process.argv[3]) : undefined;
  const backendArg = process.argv[4] ?? "cpu";
  if (backendArg !== "cpu" && backendArg !== "cuda") {
    throw new Error(`[build:icn] unsupported release backend ${backendArg}`);
  }
  await buildIcnBinary(targetArg, outDirArg, backendArg);
}
