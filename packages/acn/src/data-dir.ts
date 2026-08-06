import { homedir } from "node:os"
import { join } from "node:path"

export const defaultDataDir = (): string => join(homedir(), ".magnitude")
