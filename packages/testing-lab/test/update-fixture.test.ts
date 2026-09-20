import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { decodeUpdateConfiguration, UpdateConfiguration, UpdateRelease, ReleaseTarget, signUpdateRequest, verifyUpdateRelease } from "@magnitudedev/release/hosted-update"
import { Effect, Fiber, Option, Schedule, Schema } from "effect"
import { createHash, generateKeyPairSync, randomBytes } from "node:crypto"
import { existsSync, readFileSync } from "node:fs"
import { createRequire } from "node:module"
import { createConnection } from "node:net"
import { dirname, join } from "node:path"
import { expect, it } from "vitest"
import { Digest } from "../src/domain"
import { command, ProcessExecutorLive } from "../src/process"
import { updateFixture } from "../src/update-fixture"

const ResponseData = Schema.Struct({ status: Schema.Int, headers: Schema.Record({ key: Schema.String, value: Schema.String }), body: Schema.String })
const electronRoot = dirname(createRequire(import.meta.url).resolve("electron/package.json"))
const electron = join(electronRoot, "dist", readFileSync(join(electronRoot, "path.txt"), "utf8").trim())
const nodeFetch = `if(!process.versions.electron || process.versions.bun)throw new Error("Expected Electron Node runtime");let input='';for await(const part of process.stdin)input+=part;const request=JSON.parse(input);try{const response=await fetch(request.url,{headers:request.headers,redirect:'manual'});console.log(JSON.stringify({status:response.status,headers:Object.fromEntries(response.headers),body:Buffer.from(await response.arrayBuffer()).toString('base64')}));}catch(error){console.error(error.cause?.code ?? error.message);process.exitCode=1;}`
const request = (url: string, caPath: string, headers: Record<string, string> = {}) => Effect.gen(function* () {
  const input = yield* Schema.encode(Schema.parseJson(Schema.Struct({ url: Schema.String, headers: Schema.Record({ key: Schema.String, value: Schema.String }) })))({ url, headers })
  return yield* command(electron, ["--input-type=module", "-e", nodeFetch], {
    env: { NODE_EXTRA_CA_CERTS: caPath, ELECTRON_RUN_AS_NODE: "1" }, stdin: Option.some(input), timeoutMs: 10_000,
  })
})
const fetchFixture = (url: string, caPath: string, headers: Record<string, string> = {}) => Effect.gen(function* () {
  const result = yield* request(url, caPath, headers)
  expect(result.exitCode).toBe(0)
  return yield* Schema.decodeUnknown(Schema.parseJson(ResponseData))(result.stdout)
})
const body = (response: typeof ResponseData.Type) => Buffer.from(response.body, "base64")
const run = <A, E>(effect: Effect.Effect<A, E, FileSystem.FileSystem | import("../src/process").ProcessExecutor>) => Effect.runPromise(effect.pipe(Effect.provide([BunContext.layer, ProcessExecutorLive])))

it("serves signed private updates over trusted HTTPS and closes its owned resources", async () => {
  let owned = "", port = 0
  await run(Effect.scoped(Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const parent = yield* fs.makeTempDirectoryScoped({ prefix: "lab-update-fixture-test-" })
    const fixture = yield* updateFixture(parent)
    owned = fixture.caPath
    port = Number(new URL(fixture.origin).port)
    const configured = yield* decodeUpdateConfiguration(yield* Schema.encode(UpdateConfiguration)(fixture.configuration), true)
    const identity = generateKeyPairSync("ed25519")
    const signed = (path = "/api/update", patch: Record<string, string> = {}) => Effect.gen(function* () {
      const query = new URLSearchParams({ protocol: "1", product: "desktop", version: "1.0.0", os: "darwin", os_version: "15.0",
        arch: "arm64", package: "mac-zip", channel: "stable", ts: String(Math.floor(Date.now() / 1000)), nonce: randomBytes(16).toString("base64url"), ...patch })
      const url = new URL(`${path}?${query}`, fixture.origin)
      const authorization = yield* signUpdateRequest(identity.privateKey, url)
      return { url: url.href, headers: { authorization } }
    })
    const query = yield* signed()
    const untrusted = yield* request(query.url, "", query.headers)
    expect(untrusted.exitCode).not.toBe(0)
    expect(untrusted.stderr).toMatch(/self.signed|verify.leaf|cert/i)
    expect((yield* fetchFixture(query.url, fixture.caPath, query.headers)).status).toBe(204)
    expect((yield* fetchFixture(query.url, fixture.caPath, query.headers)).status).toBe(409)
    expect((yield* fetchFixture(query.url, fixture.caPath)).status).toBe(400)
    const expired = yield* signed("/api/update", { ts: "1" })
    expect((yield* fetchFixture(expired.url, fixture.caPath, expired.headers)).status).toBe(400)

    const content = Buffer.from("exact private update bytes")
    const path = join(parent, "candidate.zip")
    yield* fs.writeFile(path, content)
    const target = yield* Schema.decodeUnknown(ReleaseTarget)({ os: "darwin", arch: "arm64", package: "mac-zip" })
    const artifact = { path, version: "2.0.0", target, bytes: content.length, sha256: Digest.make(createHash("sha256").update(content).digest("hex")) }
    const offered = yield* fixture.publish(artifact)
    yield* fs.writeFileString(path, "source changed after admission")
    const check = yield* signed()
    const response = yield* fetchFixture(check.url, fixture.caPath, check.headers)
    expect(response.status).toBe(200)
    const received = yield* Schema.decodeUnknown(Schema.parseJson(Schema.Unknown))(body(response).toString())
    expect(yield* verifyUpdateRelease(received, target, configured.trustedPublishers)).toEqual(offered)
    const incompatible: readonly Record<string, string>[] = [{ arch: "x64" }, { version: "2.0.0" }]
    for (const patch of incompatible) {
      const check = yield* signed("/api/update", patch)
      expect((yield* fetchFixture(check.url, fixture.caPath, check.headers)).status).toBe(204)
    }
    const resolve = yield* signed("/api/download", { release: "2.0.0" })
    const redirect = yield* fetchFixture(resolve.url, fixture.caPath, resolve.headers)
    expect(redirect.status).toBe(302)
    const location = redirect.headers.location!
    expect(new URL(location).origin).toBe(fixture.origin)
    const full = yield* fetchFixture(location, fixture.caPath)
    expect(full.status).toBe(200)
    expect(body(full)).toEqual(content)
    const range = yield* fetchFixture(location, fixture.caPath, { range: "bytes=2-8", "if-range": full.headers.etag! })
    expect(range.status).toBe(206)
    expect(range.headers["content-range"]).toBe(`bytes 2-8/${content.length}`)
    expect(body(range)).toEqual(content.subarray(2, 9))
    expect((yield* fetchFixture(location, fixture.caPath, { range: "bytes=0-9999" })).status).toBe(416)
    expect((yield* fetchFixture(location, fixture.caPath, { range: "bytes=0-1", "if-range": '"wrong"' })).status).toBe(412)
    expect((yield* fetchFixture(location, fixture.caPath, { authorization: "must-not-leak" })).status).toBe(400)
    expect((yield* fixture.publish(artifact).pipe(Effect.either))._tag).toBe("Left")
    expect((yield* fetchFixture(location, fixture.caPath)).status).toBe(200)
    // A corrupt delivery still has authentic metadata and exact length. Only the owned copy changes.
    yield* fs.writeFile(path, content)
    const corrupted = yield* fixture.publish(artifact, "Corrupt")
    expect(yield* verifyUpdateRelease(corrupted, target, configured.trustedPublishers)).toEqual(offered)
    const corruptResolve = yield* signed("/api/download", { release: "2.0.0" })
    const corruptRedirect = yield* fetchFixture(corruptResolve.url, fixture.caPath, corruptResolve.headers)
    const corruptBody = body(yield* fetchFixture(corruptRedirect.headers.location!, fixture.caPath))
    expect(corruptBody.length).toBe(content.length)
    expect(corruptBody[0]).toBe(content[0]! ^ 0xff)
    expect(corruptBody.subarray(1)).toEqual(content.subarray(1))
    expect(Buffer.from(yield* fs.readFile(path))).toEqual(content)
    yield* fixture.publish(artifact)
    const recoveredResolve = yield* signed("/api/download", { release: "2.0.0" })
    const recoveredRedirect = yield* fetchFixture(recoveredResolve.url, fixture.caPath, recoveredResolve.headers)
    expect(body(yield* fetchFixture(recoveredRedirect.headers.location!, fixture.caPath))).toEqual(content)
    // Observe actual bytes in a separate native Electron/Node client before breaking its socket.
    yield* Effect.scoped(Effect.gen(function* () {
      const fault = yield* fixture.interruptDownload
      // The real segmented updater probes one byte before opening payload ranges.
      const probe = yield* fetchFixture(recoveredRedirect.headers.location!, fixture.caPath, { range: "bytes=0-0" })
      expect(probe.status).toBe(206)
      expect(body(probe)).toEqual(content.subarray(0, 1))
      expect((yield* Effect.scoped(fixture.interruptDownload).pipe(Effect.either))._tag).toBe("Left")
      const receipt = join(parent, "prefix-received")
      const interruptedFetch = `import {writeFileSync} from 'node:fs';if(!process.versions.electron||process.versions.bun)throw new Error('Native Electron required');let input='';for await(const part of process.stdin)input+=part;const data=JSON.parse(input);let bytes=0;try{const response=await fetch(data.url);for await(const part of response.body){bytes+=part.length;writeFileSync(data.receipt,String(bytes));}console.log(JSON.stringify({bytes,failed:false}));}catch{console.log(JSON.stringify({bytes,failed:true}));}`
      const input = yield* Schema.encode(Schema.parseJson(Schema.Struct({ url: Schema.String, receipt: Schema.String })))({ url: recoveredRedirect.headers.location!, receipt })
      const consumer = yield* command(electron, ["--input-type=module", "-e", interruptedFetch], {
        env: { NODE_EXTRA_CA_CERTS: fixture.caPath, ELECTRON_RUN_AS_NODE: "1" }, stdin: Option.some(input), timeoutMs: 10_000,
      }).pipe(Effect.forkScoped)
      const offered = yield* fault.started
      expect(offered.offeredBytes).toBeGreaterThan(0)
      expect(offered.offeredBytes).toBeLessThan(offered.requestedBytes)
      yield* fs.readFileString(receipt).pipe(Effect.retry(Schedule.spaced("10 millis").pipe(Schedule.intersect(Schedule.recurs(300)))))
      yield* fault.cut
      const outcome = yield* Fiber.join(consumer)
      expect(outcome.exitCode).toBe(0)
      const transferred = yield* Schema.decodeUnknown(Schema.parseJson(Schema.Struct({ bytes: Schema.Number, failed: Schema.Boolean })))(outcome.stdout)
      expect(transferred.failed).toBe(true)
      expect(transferred.bytes).toBeGreaterThan(0)
      expect(transferred.bytes).toBeLessThan(content.length)
      expect((yield* fetchFixture(recoveredRedirect.headers.location!, fixture.caPath)).status).toBe(503)
    }))
    expect(body(yield* fetchFixture(recoveredRedirect.headers.location!, fixture.caPath))).toEqual(content)
    yield* fixture.withdraw
    expect((yield* fetchFixture(location, fixture.caPath)).status).toBe(404)
    const empty = yield* signed()
    expect((yield* fetchFixture(empty.url, fixture.caPath, empty.headers)).status).toBe(204)
  })))
  expect(existsSync(owned)).toBe(false)
  const refused = await new Promise<boolean>(resolve => {
    const socket = createConnection({ host: "127.0.0.1", port })
    const done = (refused: boolean) => { socket.destroy(); resolve(refused) }
    socket.setTimeout(1000, () => done(false))
    socket.once("connect", () => done(false))
    socket.once("error", error => done((error as NodeJS.ErrnoException).code === "ECONNREFUSED"))
  })
  expect(refused).toBe(true)
}, 30_000)

it("moves private update authority between disjoint build and consumer lifetimes", async () => {
  await run(Effect.scoped(Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const parent = yield* fs.makeTempDirectoryScoped({ prefix: "lab-update-transfer-" })
    const built = yield* Effect.scoped(Effect.gen(function* () {
      const fixture = yield* updateFixture(parent)
      return { authority: fixture.authority, configuration: fixture.configuration }
    }))
    const consumer = yield* updateFixture(parent, built.authority)
    expect(consumer.configuration).toEqual(built.configuration)
    expect(yield* fs.readFileString(consumer.caPath)).toBe(built.authority.certificate)
    const target = { os: "darwin", arch: "arm64", package: "mac-zip" } as const
    const path = join(parent, "new.zip"), content = Buffer.from("transferred candidate")
    yield* fs.writeFile(path, content)
    const offered = yield* consumer.publish({ path, version: "2.0.0", target, bytes: content.length, sha256: Digest.make(createHash("sha256").update(content).digest("hex")) })
    const trusted = yield* decodeUpdateConfiguration(yield* Schema.encode(UpdateConfiguration)(built.configuration), true)
    expect(yield* verifyUpdateRelease(yield* Schema.encode(UpdateRelease)(offered), target, trusted.trustedPublishers)).toEqual(offered)
    const collision = yield* Effect.scoped(updateFixture(parent, built.authority)).pipe(Effect.either)
    expect(collision._tag).toBe("Left")
    // Failed restoration must not close the existing listener.
    const identity = generateKeyPairSync("ed25519")
    const url = new URL(`/api/update?${new URLSearchParams({ protocol: "1", product: "desktop", version: "1.0.0", os: "darwin", os_version: "15.0", arch: "arm64", package: "mac-zip", channel: "stable", ts: String(Math.floor(Date.now()/1000)), nonce: randomBytes(16).toString("base64url") })}`, consumer.origin)
    expect((yield* fetchFixture(url.href, consumer.caPath, { authorization: yield* signUpdateRequest(identity.privateKey, url) })).status).toBe(200)
    const wrongKey = generateKeyPairSync("rsa", { modulusLength: 2048 }).privateKey.export({ type: "pkcs8", format: "pem" }).toString()
    const invalid = yield* Effect.scoped(updateFixture(parent, { ...built.authority, tlsPrivateKey: wrongKey })).pipe(Effect.either)
    expect(invalid._tag).toBe("Left")
  })))
}, 30_000)
