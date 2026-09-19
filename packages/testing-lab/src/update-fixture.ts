import { FileSystem } from "@effect/platform"
import { acceptsUpdateRelease, decodeUpdateRequest, ReleaseTarget, signUpdateRelease, UpdateConfiguration, UpdateRelease, verifyUpdateRequest } from "@magnitudedev/release/hosted-update"
import { defineFSM } from "@magnitudedev/utils/fsm"
import { Clock, Effect, Option, Ref, Runtime, Schema, Stream } from "effect"
import { createHash, generateKeyPairSync, randomBytes, randomUUID } from "node:crypto"
import { join } from "node:path"
import { AssertionFailure, Digest, InfrastructureFailure } from "./domain"
import { checkedCommand } from "./process"

export const UpdateFixtureArtifact = Schema.Struct({
  path: Schema.NonEmptyString, version: Schema.NonEmptyString, target: ReleaseTarget,
  bytes: Schema.Int.pipe(Schema.positive()), sha256: Digest,
})
export const UpdateFixtureDelivery = Schema.Literal("Exact", "Corrupt")
class Empty extends Schema.TaggedClass<Empty>()("Empty", {}) {}
class Offering extends Schema.TaggedClass<Offering>()("Offering", {
  path: Schema.String, release: UpdateRelease, target: ReleaseTarget, route: Schema.String,
}) {}
const lifecycle = defineFSM({ Empty, Offering }, { Empty: ["Offering"], Offering: ["Empty", "Offering"] })
const fail = (message: string) => new InfrastructureFailure({ operation: "update-fixture", message })

/** Owns a loopback-only HTTPS origin and temporary trust; never modifies the operating-system trust store. */
export const updateFixture = (parent: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const directory = yield* fs.makeTempDirectoryScoped({ directory: parent, prefix: "update-fixture-" })
  yield* fs.chmod(directory, 0o700)
  const caPath = join(directory, "certificate.pem"), keyPath = join(directory, "tls-key.pem")
  const opensslConfig = join(directory, "openssl.cnf")
  yield* fs.writeFileString(opensslConfig, `[req]\nprompt = no\ndistinguished_name = subject\nx509_extensions = extensions\n[subject]\nCN = Magnitude isolated update fixture\n[extensions]\nbasicConstraints = critical,CA:TRUE\nkeyUsage = critical,digitalSignature,keyEncipherment,keyCertSign\nextendedKeyUsage = serverAuth\nsubjectAltName = IP:127.0.0.1,DNS:localhost\n`, { mode: 0o600 })
  yield* checkedCommand("openssl", ["req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-config", opensslConfig,
    "-keyout", keyPath, "-out", caPath], { timeoutMs: 30_000, maxOutputBytes: 64 * 1024 })
  yield* fs.chmod(keyPath, 0o600)
  const publisher = yield* Effect.try({ try: () => generateKeyPairSync("ed25519"), catch: () => fail("Could not create fixture publisher") })
  const state = yield* Ref.make<Empty | Offering>(new Empty())
  const gate = yield* Effect.makeSemaphore(1)
  // Replay admission is synchronous inside the request Effect and bounded independently of server lifetime.
  const nonces = new Map<string, number>()
  let origin = ""
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
      return new Response(Bun.file(current.path).slice(start, end + 1), {
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
  const server = yield* Effect.acquireRelease(Effect.try({ try: () => Bun.serve({ hostname: "127.0.0.1", port: 0,
    tls: { key, cert }, maxRequestBodySize: 0,
    fetch: request => Runtime.runPromise(runtime)(handler(request)),
    error: () => new Response(null, { status: 500 }),
  }), catch: () => fail("Could not bind private HTTPS fixture") }), server => Effect.promise(() => server.stop(true)))
  origin = `https://127.0.0.1:${server.port}`
  const configuration = yield* Schema.decodeUnknown(UpdateConfiguration)({ origin, acceptance: true, keyId: "lab-fixture",
    publicKey: publisher.publicKey.export({ type: "spki", format: "pem" }).toString(),
    artifactDelivery: { _tag: "PrivateAcceptance", origin },
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
  return { origin, caPath, configPath, configuration, publish, withdraw }
})
