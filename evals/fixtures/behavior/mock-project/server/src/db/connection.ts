import { drizzle } from 'drizzle-orm/bun-sqlite'
import { Database } from 'bun:sqlite'
import { config } from '../utils/config'

export const sqlite = new Database(config.dbPath)
export const db = drizzle(sqlite)