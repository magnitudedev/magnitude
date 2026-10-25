// Disposable CI fixture for the installer audit. Never publishes a release.
import { FileSystem } from '@effect/platform'
import { BunContext, BunRuntime } from '@effect/platform-bun'
import { Effect, Schema } from 'effect'
import { generateKeyPairSync, createPrivateKey } from 'node:crypto'
import { join, resolve } from 'node:path'
import { writeInstallationDistribution } from '../packages/release/scripts/build/installation-distribution'
import { writeInstallerOfferFixture } from '../packages/release/scripts/acceptance/installer-offer-fixture'
import { UpdateManifest, signUpdateManifest } from '../packages/release/src/hosted-update/manifest'
const run = Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const [mode, rootInput, artifactsInput, host] = process.argv.slice(2)
  const root = resolve(rootInput!)
  if (mode === 'prepare') {
    yield* fs.makeDirectory(root, { recursive: true })
    const keys = yield* Effect.sync(() => generateKeyPairSync('ed25519'))
    const publicKey = keys.publicKey.export({type:'spki',format:'pem'}).toString()
    yield* fs.writeFileString(join(root,'key.pem'), keys.privateKey.export({type:'pkcs8',format:'pem'}).toString(), {mode:0o600})
    yield* fs.writeFileString(join(root,'configuration.json'),JSON.stringify({origin:'https://localhost:18443',keyId:'installer-audit',publicKey}))
    return
  }
  const artifacts = resolve(artifactsInput!)
  const config = JSON.parse(yield* fs.readFileString(join(root,'configuration.json')))
  const key = createPrivateKey(yield* fs.readFileString(join(root,'key.pem')))
  const version = JSON.parse(yield* fs.readFileString('packages/launcher/package.json')).version
  const arch = host!.includes('arm64') ? 'arm64' : 'x64'
  const os = host!.startsWith('darwin') ? 'darwin' : 'linux'
  const hosting = join(root,'hosting')
  yield* fs.makeDirectory(hosting)
  for (const name of yield* fs.readDirectory(artifacts)) {
    if (!name.endsWith('.artifact.json')) continue
    const artifact = JSON.parse(yield* fs.readFileString(join(artifacts,name)))
    if (os === 'darwin' ? artifact.kind !== 'desktop-update' : artifact.kind !== 'desktop-installer') continue
    const format = os === 'darwin' ? 'mac-zip' : artifact.filename.endsWith('.deb') ? 'deb' : artifact.filename.endsWith('.rpm') ? 'rpm' : 'pacman'
    const destination = join(hosting,format)
    yield* writeInstallationDistribution({output:destination,origin:'https://localhost:18443',appleTeam:process.env.APPLE_TEAM_ID || '8YCG6CT28C',windowsPublisher:'Magnitude Update Acceptance',publicKey:config.publicKey})
    const manifest = yield* Schema.decodeUnknown(UpdateManifest)({protocol:1,version,tag:`@magnitudedev/cli@${version}`,commit:process.env.MAGNITUDE_SOURCE_COMMIT,artifact:{id:artifact.id,target:{os,arch,package:format},filename:artifact.filename,bytes:artifact.bytes,sha256:artifact.sha256}})
    yield* writeInstallerOfferFixture(destination,yield* signUpdateManifest(manifest,key),config.publicKey)
    const target = join(destination,'magnitudedev/magnitude/releases/download',manifest.tag)
    yield* fs.makeDirectory(target,{recursive:true})
    yield* fs.copyFile(join(artifacts,artifact.filename),join(target,artifact.filename))
  }
})
BunRuntime.runMain(run.pipe(Effect.provide(BunContext.layer)))
