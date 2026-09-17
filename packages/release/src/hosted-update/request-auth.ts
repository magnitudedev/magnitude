import { createHash, createPublicKey, randomBytes, sign, verify, type KeyObject } from "node:crypto"
import { Effect, Schema } from "effect"

export const InstallationId = Schema.String.pipe(Schema.pattern(/^[a-f0-9]{64}$/), Schema.brand("InstallationId"))
export type InstallationId = typeof InstallationId.Type
export const RequestNonce = Schema.String.pipe(Schema.pattern(/^[A-Za-z0-9_-]{21}[AQgw]$/), Schema.brand("UpdateRequestNonce"))
export type RequestNonce = typeof RequestNonce.Type
export class InvalidUpdateSignature extends Schema.TaggedError<InvalidUpdateSignature>()("InvalidUpdateSignature", {}) {}
export class UpdateSigningFailed extends Schema.TaggedError<UpdateSigningFailed>()("UpdateSigningFailed", {}) {}

// SSH's Ed25519 public blob is string("ssh-ed25519") + string(raw public key).
const sshPrefix = Buffer.from("0000000b7373682d6564323535313900000020", "hex")
const spkiPrefix = Buffer.from("302a300506032b6570032100", "hex")
const publicBytes = (key: KeyObject): Buffer => {
  const publicKey = key.type === "public" ? key : createPublicKey(key)
  if (publicKey.asymmetricKeyType !== "ed25519") throw new Error("Expected Ed25519")
  const der = publicKey.export({ type: "spki", format: "der" })
  if (der.length !== spkiPrefix.length + 32 || !der.subarray(0, spkiPrefix.length).equals(spkiPrefix)) throw new Error("Invalid public key")
  return der.subarray(spkiPrefix.length)
}

/** Match Go url.Values.Encode, including the characters encodeURIComponent leaves unescaped. */
const escapeQuery = (value: string) => encodeURIComponent(value).replace(/[!'()*]/g, value => `%${value.charCodeAt(0).toString(16).toUpperCase()}`).replace(/%20/g, "+")
export const updateQuery = (fields: Readonly<Record<string, string>>): string => Object.keys(fields).sort()
  .map(key => `${escapeQuery(key)}=${escapeQuery(fields[key]!)}`).join("&")
export const newUpdateNonce = Effect.sync(() => RequestNonce.make(randomBytes(16).toString("base64url")))
export const installationId = (key: KeyObject) => Effect.try({
  try: () => InstallationId.make(createHash("sha256").update(publicBytes(key)).digest("hex")),
  catch: () => new UpdateSigningFailed(),
})

/** Ollama Authorization: base64(SSH public key):base64(raw Ed25519 signature). */
export const signUpdateRequest = (key: KeyObject, url: URL) => Effect.try({
  try: () => {
    const signature = sign(null, Buffer.from(`GET,${url.pathname}${url.search}`), key)
    return `${Buffer.concat([sshPrefix, publicBytes(key)]).toString("base64")}:${signature.toString("base64")}`
  }, catch: () => new UpdateSigningFailed(),
})

const decodeBase64 = (value: string): Buffer => {
  const decoded = Buffer.from(value, "base64")
  if (decoded.toString("base64") !== value) throw new Error("Noncanonical base64")
  return decoded
}

/** Route/origin, field validation, timestamp admission and nonce replay checks belong to the server. */
export const verifyUpdateRequest = (authorization: string, url: URL) => Effect.try({
  try: () => {
    if (authorization.length > 256 || url.href.length > 4096) throw new Error("Oversized request")
    const parts = authorization.split(":")
    if (parts.length !== 2) throw new Error("Invalid authorization")
    const blob = decodeBase64(parts[0]!)
    const signature = decodeBase64(parts[1]!)
    if (blob.length !== sshPrefix.length + 32 || !blob.subarray(0, sshPrefix.length).equals(sshPrefix) || signature.length !== 64) throw new Error("Invalid key or signature")
    const raw = blob.subarray(sshPrefix.length)
    const key = createPublicKey({ key: Buffer.concat([spkiPrefix, raw]), format: "der", type: "spki" })
    if (!verify(null, Buffer.from(`GET,${url.pathname}${url.search}`), key, signature)) throw new Error("Invalid signature")
    return InstallationId.make(createHash("sha256").update(raw).digest("hex"))
  }, catch: () => new InvalidUpdateSignature(),
})
