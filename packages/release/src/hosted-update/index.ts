export { checkHostedUpdate, resolveHostedDownload, HostedUpdateCheckFailed, UpdateClientMetadata, type HostedUpdateConnection } from "./client"
export { signUpdateRequest, UpdateSigningFailed } from "./request-auth"
export { downloadUpdateArtifact } from "./installer-download"
export { UpdateManifest, UpdateCandidate, SignedUpdateManifest, acceptsUpdateManifest, verifyUpdateManifest, decodePublisherPublicKey } from "./manifest"
