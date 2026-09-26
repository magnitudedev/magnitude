import {
  IcnBinaryIdentity,
  IcnInstallationDeclaration,
  IcnStartupRecord,
} from "@magnitudedev/icn-protocol"
import { ICN_EXECUTABLE_NAME } from "@magnitudedev/release/executables"
import { ProcessGroupControllerLive } from "@magnitudedev/utils/process-groups/native"
import { mergeWindowsEnvironment } from "@magnitudedev/utils/windows-native"
import { Effect, Option, Schema } from "effect"
import { mkdir, mkdtemp, readFile, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { delimiter, dirname, resolve } from "node:path"

/**
 * The environment a managed parent gives the service: Windows resolves the owned CRT from
 * `runtime/`; Unix loaders ignore inherited search paths so only owned loader paths apply.
 */
export const installationEnvironment = (
  root: string,
  platform: NodeJS.Platform = process.platform,
): Readonly<Record<string, string | undefined>> =>
  platform === "win32"
    ? mergeWindowsEnvironment(process.env, {
      PATH: [resolve(root, "runtime"), process.env.PATH].filter(Boolean).join(delimiter),
    })
    : { ...process.env, ...(platform === "darwin" ? { DYLD_LIBRARY_PATH: "" } : { LD_LIBRARY_PATH: "" }) }

const HealthBody = Schema.Struct({ ready: Schema.Literal(true), instanceId: Schema.String })

const within = async <T>(operation: Promise<T>, milliseconds: number): Promise<T | undefined> => {
  let timer: ReturnType<typeof setTimeout> | undefined
  try {
    return await Promise.race([
      operation,
      new Promise<undefined>((resolve) => {
        timer = setTimeout(() => resolve(undefined), milliseconds)
      }),
    ])
  } finally {
    if (timer !== undefined) clearTimeout(timer)
  }
}

/** `serve` on an ephemeral port: handshake, health, authenticated hardware, then parent-loss exit. */
const smokeServe = async (
  binary: string,
  installation: string,
  environment: Readonly<Record<string, string | undefined>>,
): Promise<void> => {
  const scratch = await mkdtemp(resolve(tmpdir(), "magnitude-inference-smoke-"))
  const modelStore = resolve(scratch, "model-store")
  const cacheRoot = resolve(scratch, "cache")
  await Promise.all([
    mkdir(modelStore, { recursive: true, mode: 0o700 }),
    mkdir(cacheRoot, { recursive: true, mode: 0o700 }),
  ])
  const token = crypto.randomUUID()
  const instance = `inference-smoke-${crypto.randomUUID()}`
  const child = Bun.spawn([
    binary,
    "serve",
    "--bind",
    "127.0.0.1:0",
    "--instance-id",
    instance,
    "--exit-on-stdin-eof",
    "--installation",
    installation,
    "--model-store",
    modelStore,
    "--cache-root",
    cacheRoot,
  ], {
    cwd: scratch,
    // Managed inference lifetime guards require it to lead its own process group.
    detached: process.platform !== "win32",
    env: { ...environment, MAGNITUDE_ICN_AUTH_TOKEN: token },
    stdin: "pipe",
    stdout: "pipe",
    stderr: "inherit",
  })
  let reaped = false
  try {
    const reader = child.stdout.getReader()
    const decoder = new TextDecoder()
    const readiness = async (): Promise<{ readonly origin: string }> => {
      let pending = ""
      while (pending.length <= 64 * 1024) {
        const next = await reader.read()
        if (next.done) break
        pending += decoder.decode(next.value, { stream: true })
        let newline = pending.indexOf("\n")
        while (newline >= 0) {
          const line = pending.slice(0, newline).trimEnd()
          pending = pending.slice(newline + 1)
          const prefix = "MAGNITUDE_ICN_READY "
          if (line.startsWith(prefix)) {
            const value = Schema.decodeUnknownSync(Schema.parseJson(IcnStartupRecord))(line.slice(prefix.length))
            if (value.instanceId !== instance) {
              throw new Error("inference readiness record has the wrong identity")
            }
            return { origin: value.origin }
          }
          newline = pending.indexOf("\n")
        }
      }
      throw new Error("inference exited without a bounded readiness record")
    }
    const ready = await within(readiness(), 30_000)
    if (ready === undefined) throw new Error("inference startup smoke timed out")
    const authorization = { authorization: `Bearer ${token}` }
    const health = await fetch(`${ready.origin}/health`, { headers: authorization, signal: AbortSignal.timeout(10_000) })
    if (!health.ok) throw new Error(`inference health returned HTTP ${health.status}`)
    if (Schema.decodeUnknownSync(HealthBody)(await health.json()).instanceId !== instance) {
      throw new Error("inference health returned the wrong identity")
    }
    const anonymous = await fetch(`${ready.origin}/api/v1/hardware`, { signal: AbortSignal.timeout(10_000) })
    if (anonymous.status !== 401) {
      throw new Error(`inference hardware without the owner token returned HTTP ${anonymous.status}`)
    }
    const hardware = await fetch(`${ready.origin}/api/v1/hardware`, { headers: authorization, signal: AbortSignal.timeout(10_000) })
    if (!hardware.ok) {
      throw new Error(`inference authenticated hardware returned HTTP ${hardware.status}`)
    }
    const owned = process.platform === "win32"
      ? Option.none()
      : await Effect.runPromise(ProcessGroupControllerLive.inspect(child.pid))
    if (process.platform !== "win32" && Option.isNone(owned)) {
      throw new Error("inference disappeared before parent-loss acceptance")
    }
    child.stdin.end()
    const exitCode = await within(child.exited, 5_000)
    if (exitCode === undefined) {
      throw new Error("inference did not exit after its managed parent pipe closed")
    }
    reaped = true
    // EOF means owner loss, so the watchdog kills its group rather than exiting gracefully.
    if (exitCode !== 91 && child.signalCode !== "SIGKILL") {
      throw new Error(`inference parent-loss watchdog exited with unexpected code ${exitCode} and signal ${child.signalCode}`)
    }
    if (Option.isSome(owned) && !await Effect.runPromise(ProcessGroupControllerLive.waitForGroupExit({ leader: owned.value }, "5 seconds"))) {
      throw new Error("inference parent-loss watchdog left process-group members alive")
    }
  } finally {
    if (!reaped) {
      child.kill("SIGTERM")
      child.stdin.end()
      const exited = await within(child.exited.then(() => true), 5_000)
      if (!exited) {
        child.kill("SIGKILL")
        await child.exited
      }
    }
    await child[Symbol.asyncDispose]()
    await rm(scratch, { recursive: true, force: true })
  }
}

/**
 * Smokes one installation layout end to end through the unchanged ACN launch interface:
 * the identity probe must match the declaration's `nativeBuild`, then `serve` must complete
 * the handshake, answer health and authenticated hardware, and exit on stdin EOF.
 */
export const smokeInstallation = async (installation: string): Promise<IcnBinaryIdentity> => {
  const declaration = Schema.decodeUnknownSync(Schema.parseJson(IcnInstallationDeclaration))(
    await readFile(installation, "utf8"),
  )
  const root = dirname(installation)
  const binary = resolve(root, "bin", `${ICN_EXECUTABLE_NAME}${process.platform === "win32" ? ".exe" : ""}`)
  const environment = installationEnvironment(root)
  const probe = Bun.spawn([binary, "version", "--json"], { env: environment, stdout: "pipe", stderr: "pipe" })
  const [code, stdout, stderr] = await Promise.all([
    probe.exited,
    new Response(probe.stdout).text(),
    new Response(probe.stderr).text(),
  ])
  if (code !== 0) throw new Error(`inference identity probe exited with ${code}: ${stderr.trim()}`)
  const identity = Schema.decodeUnknownSync(Schema.parseJson(IcnBinaryIdentity))(stdout)
  if (identity.native_build !== declaration.nativeBuild) {
    throw new Error(`inference identity ${identity.native_build} differs from the declared ${declaration.nativeBuild}`)
  }
  await smokeServe(binary, installation, environment)
  return identity
}

if (import.meta.main) {
  const installation = process.argv[2]
  if (!installation) throw new Error("usage: smoke.ts <installation.json>")
  const started = performance.now()
  const identity = await smokeInstallation(resolve(installation))
  console.log(`inference ${identity.version} (${identity.native_build}) passed the installation smoke in ${Math.round(performance.now() - started)} ms`)
}
