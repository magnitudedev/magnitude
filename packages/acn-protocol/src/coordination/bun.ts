import { Layer } from "effect"
import { BunSqliteDriver } from "./bun-sqlite-driver"
import { SqliteDriver } from "./sqlite-driver"

export { BunSqliteDriver }

export const BunSqliteDriverLayer = Layer.succeed(SqliteDriver, BunSqliteDriver)
