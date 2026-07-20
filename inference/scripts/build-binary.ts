import {
  copyFile,
  mkdir,
  mkdtemp,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { $ } from "bun";
import {
  getDefaultBunTarget,
  getTargetInfo,
} from "../../scripts/release-target";

const PROJECT_ROOT = resolve(import.meta.dir, "../..");

function rustTarget(target: string): string {
  const { platform, arch } = getTargetInfo(target);
  const mapped: Record<string, string> = {
    "darwin-arm64": "aarch64-apple-darwin",
    "darwin-x64": "x86_64-apple-darwin",
    "linux-arm64": "aarch64-unknown-linux-musl",
    "linux-x64": "x86_64-unknown-linux-musl",
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

async function smokeIcn(binaryFile: string): Promise<void> {
  const version = Bun.spawn([binaryFile, "version", "--json"], {
    stdout: "pipe",
    stderr: "pipe",
  });
  const identity = JSON.parse(await new Response(version.stdout).text());
  if ((await version.exited) !== 0 || identity.api_version !== 1) {
    throw new Error("[build:icn] version smoke failed");
  }

  const doctor = Bun.spawn([binaryFile, "doctor"], {
    stdout: "pipe",
    stderr: "pipe",
  });
  const doctorOutput = await new Response(doctor.stdout).text();
  if ((await doctor.exited) !== 0 || !doctorOutput.includes("successfully")) {
    throw new Error("[build:icn] doctor smoke failed");
  }

  const modelStore = await mkdtemp(join(tmpdir(), "magnitude-icn-smoke-"));
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
  outDir = resolve(PROJECT_ROOT, "release")
): Promise<string> {
  const info = getTargetInfo(target);
  const cargoTarget = rustTarget(target);
  const binDir = resolve(PROJECT_ROOT, "bin");
  const manifestPath = resolve(PROJECT_ROOT, "inference/Cargo.toml");
  const name = "magnitude-icn" + info.executableExt;
  const companionManifestName = "magnitude-icn-manifest.json";
  const tarballPath = resolve(
    outDir,
    `magnitude-icn-${info.platformKey}.tar.gz`
  );

  await mkdir(binDir, { recursive: true });
  await mkdir(outDir, { recursive: true });
  console.log("[build:icn] Building " + cargoTarget);
  await $`cargo build --release --manifest-path ${manifestPath} -p icn-server --target ${cargoTarget}`;

  const source = resolve(
    PROJECT_ROOT,
    "inference/target",
    cargoTarget,
    "release",
    name
  );
  const destination = resolve(binDir, name);
  await copyFile(source, destination);
  if (info.platform === "darwin") {
    await $`codesign --force --deep --sign - ${destination}`;
  }
  if (target === getDefaultBunTarget()) await smokeIcn(destination);

  const pin = await readFile(
    resolve(PROJECT_ROOT, "inference/native-pin.toml"),
    "utf8"
  );
  const revision = (table: string) => {
    const match = pin.match(
      new RegExp(`\\[${table}\\][\\s\\S]*?revision\\s*=\\s*"([^"]+)"`)
    );
    if (!match)
      throw new Error(
        `[build:icn] missing ${table}.revision in native-pin.toml`
      );
    return match[1];
  };
  const nativeBuild = `bindings:${revision(
    "llama_cpp_rs"
  )};llama_cpp:${revision("llama_cpp")}`;
  await writeFile(
    resolve(binDir, companionManifestName),
    JSON.stringify(
      {
        schemaVersion: 1,
        binary: name,
        sha256: await fileSha256(destination),
        apiVersion: 1,
        nativeBuild,
        target: cargoTarget,
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
  await buildIcnBinary(targetArg, outDirArg);
}
