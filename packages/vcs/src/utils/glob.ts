import { Glob } from "bun"

/** Create a reusable glob matcher using Bun's native Glob implementation. */
export function createGlobMatcher(pattern: string): (path: string) => boolean {
  const glob = new Glob(pattern)
  return (path: string) => glob.match(path)
}
