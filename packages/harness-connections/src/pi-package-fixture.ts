import { packageContentFingerprint, sha256 } from "@magnitudedev/release/plugin-content"
import { MAGNITUDE_RPC_VERSION } from "@magnitudedev/sdk"
import packageManifest from "../../../integrations/pi/package.json"

const manifest = { name: packageManifest.name, version: packageManifest.version, type: "module", files: ["dist", "README.md"], pi: { extensions: ["./dist/magnitude.js"] }, dependencies: {}, peerDependencies: {} }
const contents = { "dist/magnitude.js": "export default () => {}\n", "README.md": "Test Pi package\n" }
const hashes = Object.fromEntries(Object.entries(contents).map(([path,contents]) => [path,sha256(contents)]))
export const fixtureMetadata = { name: manifest.name, version: manifest.version, rpcVersion: MAGNITUDE_RPC_VERSION, contentFingerprint: packageContentFingerprint(manifest,MAGNITUDE_RPC_VERSION,hashes), files: hashes }
export const fixtureSelection = { host: "pi", ...fixtureMetadata, filename: "pi.tgz", integrity: "sha512-test" }
export const fixturePackageFiles = {
  ...contents,
  "package.json": `${JSON.stringify(manifest)}\n`,
  "dist/magnitude-plugin.json": `${JSON.stringify(fixtureMetadata)}\n`,
}
export const piPackageFiles = (root: string) => Object.fromEntries(Object.entries(fixturePackageFiles).map(([path,contents]) => [`${root}/${path}`,contents]))
