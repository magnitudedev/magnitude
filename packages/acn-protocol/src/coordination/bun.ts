import { Layer } from "effect"
import { BunSqliteMutex } from "./bun-sqlite-mutex"
import { SqliteMutex } from "./sqlite-mutex"

export { BunSqliteMutex }

export const BunSqliteMutexLayer = Layer.succeed(SqliteMutex, BunSqliteMutex)
