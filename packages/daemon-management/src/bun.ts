import { Layer } from "effect"
import { BunSqliteDriver } from "./bun-sqlite-driver"
import { SqliteDriver } from "./sqlite-driver"

export { BunSqliteDriver }

export const BunSqliteDriverLayer = Layer.succeed(SqliteDriver, BunSqliteDriver)

// Static path expression is intentional: Bun embeds the target's native addon at compilation.
// Loading stays lazy so passive CLI help/version commands do not initialize native adapters.
import { nativeHostLayerFromLoader } from "./desktop-native/index"
import { windowsProcessObserverLayerFromLoader } from "./desktop-native/windows-process-observer"
const loadBundledNative = () => require("../dist/native/" + process.platform + "-" + process.arch + "/desktop-host.node")
export const bundledWindowsNative = {
  host: nativeHostLayerFromLoader(loadBundledNative),
  observer: windowsProcessObserverLayerFromLoader(loadBundledNative),
}
