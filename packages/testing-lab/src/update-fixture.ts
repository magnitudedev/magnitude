import { FileSystem } from "@effect/platform"
import { acceptsUpdateRelease, decodeUpdateRequest, ReleaseTarget, signUpdateRelease, UpdateConfiguration, UpdateRelease, verifyUpdateRequest } from "@magnitudedev/release/hosted-update"
import { defineFSM } from "@magnitudedev/utils/fsm"
import { Clock, Deferred, Effect, Option, Ref, Runtime, Schema, Stream } from "effect"
import { createHash, createPrivateKey, createPublicKey, generateKeyPairSync, randomBytes, randomUUID, X509Certificate } from "node:crypto"
import { join } from "node:path"
import { AssertionFailure, Digest, InfrastructureFailure } from "./domain"
import { checkedCommand } from "./process"

export const UpdateFixtureArtifact = Schema.Struct({
  path: Schema.NonEmptyString, version: Schema.NonEmptyString, target: ReleaseTarget,
  bytes: Schema.Int.pipe(Schema.positive()), sha256: Digest,
})
/** Private run material, stored as an authorized object, never as result evidence. */
export const UpdateFixtureAuthority = Schema.Struct({ schemaVersion: Schema.Literal(1),
  origin: Schema.String.pipe(Schema.pattern(/^https:\/\/127\.0\.0\.1:[1-9][0-9]{3,4}$/)),
  certificate: Schema.NonEmptyString, tlsPrivateKey: Schema.NonEmptyString, publisherPrivateKey: Schema.NonEmptyString,
})
export type UpdateFixtureAuthority = typeof UpdateFixtureAuthority.Type
export const InterruptedUpdateTransfer = Schema.Struct({ offeredBytes: Schema.Int.pipe(Schema.positive()), requestedBytes: Schema.Int.pipe(Schema.positive()) })
export const UpdateFixtureDelivery = Schema.Literal("Exact", "Corrupt")
class Empty extends Schema.TaggedClass<Empty>()("Empty", {}) {}
class Offering extends Schema.TaggedClass<Offering>()("Offering", {
  path: Schema.String, release: UpdateRelease, target: ReleaseTarget, route: Schema.String,
}) {}
const lifecycle = defineFSM({ Empty, Offering }, { Empty: ["Offering"], Offering: ["Empty", "Offering"] })
const fail = (message: string) => new InfrastructureFailure({ operation: "update-fixture", message })

/** Owns a loopback-only HTTPS origin and temporary trust; never modifies the operating-system trust store. */
export const updateFixture = (parent: string, restored?: UpdateFixtureAuthority) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const directory = yield* fs.makeTempDirectoryScoped({ directory: parent, prefix: "update-fixture-" })
  yield* fs.chmod(directory, 0o700)
  const caPath = join(directory, "certificate.pem"), keyPath = join(directory, "tls-key.pem")
  if (restored) yield* Schema.decodeUnknown(UpdateFixtureAuthority)(restored).pipe(Effect.mapError(() => fail("Malformed private update authority")))
  const opensslConfig = join(directory, "openssl.cnf")
  if (restored) {
    yield* Effect.try({ try: () => {
      const certificate = new X509Certificate(restored.certificate)
      if (!certificate.checkPrivateKey(createPrivateKey(restored.tlsPrivateKey)) || certificate.checkIP("127.0.0.1") !== "127.0.0.1"
        || Date.parse(certificate.validTo) <= Date.now() || Date.parse(certificate.validFrom) > Date.now()
        || Number(new URL(restored.origin).port) > 65535) throw new Error("Invalid fixture authority")
    }, catch: () => fail("Restored update authority has invalid or expired loopback TLS material") })
    yield* fs.writeFileString(caPath, restored.certificate, { mode: 0o600 })
    yield* fs.writeFileString(keyPath, restored.tlsPrivateKey, { mode: 0o600 })
  } else {
    yield* fs.writeFileString(opensslConfig, `[req]\nprompt = no\ndistinguished_name = subject\nx509_extensions = extensions\n[subject]\nCN = Magnitude isolated update fixture\n[extensions]\nbasicConstraints = critical,CA:TRUE\nkeyUsage = critical,digitalSignature,keyEncipherment,keyCertSign\nextendedKeyUsage = serverAuth\nsubjectAltName = IP:127.0.0.1,DNS:localhost\n`, { mode: 0o600 })
    yield* checkedCommand("openssl", ["req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-config", opensslConfig,
      "-keyout", keyPath, "-out", caPath], { timeoutMs: 30_000, maxOutputBytes: 64 * 1024 })
    yield* fs.chmod(keyPath, 0o600)
  }
  const publisher = yield* Effect.try({ try: () => {
    const privateKey = restored ? createPrivateKey(restored.publisherPrivateKey) : generateKeyPairSync("ed25519").privateKey
    if (privateKey.asymmetricKeyType !== "ed25519") throw new Error("Invalid fixture signing key")
    return { privateKey, publicKey: createPublicKey(privateKey) }
  }, catch: () => fail("Could not create fixture publisher") })
  const state = yield* Ref.make<Empty | Offering>(new Empty())
  const gate = yield* Effect.makeSemaphore(1)
  // Replay admission is synchronous inside the request Effect and bounded independently of server lifetime.
  const nonces = new Map<string, number>()
  let origin = ""
  type Fault = { cut: boolean; controllers: Set<ReadableStreamDefaultController<Uint8Array>>; started: Deferred.Deferred<typeof InterruptedUpdateTransfer.Type> }
  let activeFault: Fault | undefined
  const handler = (request: Request) => Effect.gen(function* () {
    const url = new URL(request.url)
    if (url.origin !== origin || request.method !== "GET") return new Response(null, { status: 400 })
    const current = yield* Ref.get(state)
    if (current._tag === "Offering" && url.pathname === current.route && !url.search) {
      if (request.headers.has("authorization") || request.headers.has("cookie")) return new Response(null, { status: 400 })
      const etag = `"${current.release.sha256}"`
      const range = request.headers.get("range")
      let start = 0, end = current.release.bytes - 1
      if (range) {
        const match = /^bytes=(\d+)-(\d+)$/.exec(range)
        if (!match) return new Response(null, { status: 416 })
        start = Number(match[1]); end = Number(match[2])
        if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start > end || end >= current.release.bytes) return new Response(null, { status: 416 })
        if (request.headers.has("if-range") && request.headers.get("if-range") !== etag) return new Response(null, { status: 412 })
      }
      const fault = activeFault
      if (fault?.cut) return new Response(null, { status: 503, headers: { "cache-control": "no-store" } })
      let body: Blob | ReadableStream<Uint8Array> = Bun.file(current.path).slice(start, end + 1)
      if (fault) {
        const requestedBytes = end - start + 1
        if (requestedBytes < 2) return new Response(null, { status: 503 })
        const offeredBytes = Math.min(64 * 1024, requestedBytes - 1)
        const prefix = yield* Effect.tryPromise({ try: () => Bun.file(current.path).slice(start, start + offeredBytes).bytes(), catch: () => fail("Cannot read interrupted update prefix") })
        let owner: ReadableStreamDefaultController<Uint8Array> | undefined
        body = new ReadableStream<Uint8Array>({
          start(controller) {
            owner = controller
            if (fault.cut) { controller.error(new Error("Test update connection interrupted")); return }
            fault.controllers.add(controller)
            controller.enqueue(prefix)
            Runtime.runSync(runtime)(Deferred.succeed(fault.started, { offeredBytes, requestedBytes }))
          },
          cancel() { if (owner) fault.controllers.delete(owner) },
        })
      }
      return new Response(body, {
        status: range ? 206 : 200, headers: {
          "content-length": String(end - start + 1), "content-type": "application/octet-stream", etag,
          "cache-control": "no-store", "accept-ranges": "bytes",
          ...(range ? { "content-range": `bytes ${start}-${end}/${current.release.bytes}` } : {}),
        },
      })
    }
    if (url.pathname !== "/api/update" && url.pathname !== "/api/download") return new Response(null, { status: 404 })
    const now = Math.floor((yield* Clock.currentTimeMillis) / 1000)
    const identity = yield* verifyUpdateRequest(request.headers.get("authorization") ?? "", url)
    const metadataUrl = new URL(url)
    const release = metadataUrl.searchParams.getAll("release")
    if (url.pathname === "/api/download") {
      if (release.length !== 1) return new Response(null, { status: 400 })
      metadataUrl.searchParams.delete("release")
    }
    const metadata = yield* decodeUpdateRequest(metadataUrl, now)
    for (const [nonce, expiry] of nonces) if (expiry < now) nonces.delete(nonce)
    const nonce = `${identity}:${metadata.nonce}`
    if (nonces.has(nonce)) return new Response(null, { status: 409 })
    if (nonces.size >= 4096) return new Response(null, { status: 429 })
    nonces.set(nonce, metadata.ts + 300)
    const available = current._tag === "Offering" && current.target.os === metadata.os && current.target.arch === metadata.arch
      && current.target.package === metadata.package && acceptsUpdateRelease(current.release, metadata.version)
    if (url.pathname === "/api/update") return available
      ? new Response(yield* Schema.encode(Schema.parseJson(UpdateRelease))(current.release), { headers: { "content-type": "application/json", "cache-control": "no-store" } })
      : new Response(null, { status: 204 })
    if (!available || release[0] !== current.release.version) return new Response(null, { status: 409 })
    return new Response(null, { status: 302, headers: { location: `${origin}${current.route}`, "cache-control": "no-store" } })
  }).pipe(Effect.catchAll(() => Effect.succeed(new Response(null, { status: 400 }))), Effect.timeoutOption("10 seconds"),
    Effect.map(value => value._tag === "Some" ? value.value : new Response(null, { status: 504 })))
  const runtime = yield* Effect.runtime<never>()
  const key = yield* fs.readFileString(keyPath), cert = yield* fs.readFileString(caPath)
  const server = yield* Effect.acquireRelease(Effect.try({ try: () => Bun.serve({ hostname: "127.0.0.1", port: restored ? Number(new URL(restored.origin).port) : 0,
    tls: { key, cert }, maxRequestBodySize: 0,
    fetch: request => Runtime.runPromise(runtime)(handler(request)),
    error: () => new Response(null, { status: 500 }),
  }), catch: () => fail("Could not bind private HTTPS fixture") }), server => Effect.promise(() => server.stop(true)))
  origin = `https://127.0.0.1:${server.port}`
  const configuration = yield* Schema.decodeUnknown(UpdateConfiguration)({ origin, acceptance: true, keyId: "lab-fixture",
    publicKey: publisher.publicKey.export({ type: "spki", format: "pem" }).toString(),
    artifactDelivery: { _tag: "PrivateAcceptance", origin },
    ...(process.platform === "win32" ? { windowsPublisher: "Magnitude Update Acceptance" } : {}),
  })
  const configPath = join(directory, "configuration.json")
  yield* fs.writeFileString(configPath, yield* Schema.encode(Schema.parseJson(UpdateConfiguration))(configuration), { mode: 0o600 })
  const publish = (artifact: typeof UpdateFixtureArtifact.Type, delivery: typeof UpdateFixtureDelivery.Type = "Exact") => gate.withPermits(1)(Effect.gen(function* () {
    const checked = yield* Schema.decodeUnknown(UpdateFixtureArtifact)(artifact)
    const behavior = yield* Schema.decodeUnknown(UpdateFixtureDelivery)(delivery)
    const path = join(directory, `${randomUUID()}.package`)
    yield* fs.copyFile(checked.path, path)
    yield* fs.chmod(path, 0o600)
    const hash = createHash("sha256")
    let bytes = 0
    yield* fs.stream(path).pipe(Stream.runForEach(chunk => Effect.sync(() => { hash.update(chunk); bytes += chunk.byteLength })))
    if (bytes !== checked.bytes || hash.digest("hex") !== checked.sha256) return yield* new AssertionFailure({ message: "Update fixture artifact does not match admitted bytes" })
    const release = yield* signUpdateRelease({ version: checked.version, bytes, sha256: checked.sha256 }, checked.target, publisher.privateKey)
    // Fault injection preserves valid signed metadata and length, changing only owned delivery bytes.
    // Admission always checks the original source first; corrupt input cannot masquerade as a fixture.
    if (behavior === "Corrupt") yield* Effect.scoped(Effect.gen(function* () {
      const file = yield* fs.open(path, { flag: "r+" })
      const first = yield* file.readAlloc(1)
      if (Option.isNone(first)) return yield* new AssertionFailure({ message: "Update fixture copy is empty" })
      yield* file.seek(0, "start")
      yield* file.write(new Uint8Array([first.value[0]! ^ 0xff]))
    }))
    yield* Ref.update(state, current => lifecycle.transition(current, "Offering", {
      path, release, target: checked.target, route: `/artifacts/${randomBytes(32).toString("hex")}/package`,
    }))
    return release
  }))
  const withdraw = gate.withPermits(1)(Ref.update(state, current => current._tag === "Empty" ? current : lifecycle.transition(current, "Empty", {})))
  /** Fault scope is exclusive and affects only this fixture's archive responses, never host networking. */
  const interruptDownload = Effect.gen(function* () {
    const started = yield* Deferred.make<typeof InterruptedUpdateTransfer.Type>()
    const fault: Fault = { cut: false, controllers: new Set(), started }
    const cut = Effect.sync(() => {
      fault.cut = true
      for (const controller of fault.controllers) { try { controller.error(new Error("Test update connection interrupted")) } catch { /* Client already closed. */ } }
      fault.controllers.clear()
    })
    yield* Effect.acquireRelease(Effect.suspend(() => {
      if (activeFault) return Effect.fail(fail("An update transfer fault is already active"))
      activeFault = fault
      return Effect.void
    }), () => cut.pipe(Effect.zipRight(Effect.sync(() => { if (activeFault === fault) activeFault = undefined }))))
    return { started: Deferred.await(started).pipe(Effect.timeoutFail({ duration: "30 seconds", onTimeout: () => fail("App did not start an update archive transfer") })), cut }
  })
  const authority = UpdateFixtureAuthority.make({ schemaVersion: 1, origin, certificate: cert, tlsPrivateKey: key,
    publisherPrivateKey: publisher.privateKey.export({ type: "pkcs8", format: "pem" }).toString() })
  return { origin, caPath, configPath, configuration, publish, withdraw, authority, interruptDownload }
})
