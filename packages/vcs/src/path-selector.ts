import type { Delta, RestoreScope } from "./types"
import { createGlobMatcher } from "./utils/glob"

function normalizeSelector(selector: string): string {
  return selector.trim().replace(/^\.\/+/, "")
}

function hasGlobMagic(selector: string): boolean {
  return /[*?[\]{}]/.test(selector)
}

export function selectorToRestoreScope(selector: string): RestoreScope {
  const normalized = normalizeSelector(selector)
  if (hasGlobMagic(normalized)) {
    return { kind: "glob", pattern: normalized }
  }
  if (normalized.endsWith("/")) {
    return { kind: "directory", path: normalized.slice(0, -1) }
  }
  return { kind: "file", path: normalized }
}

export function createRestoreScopePredicate(scope: RestoreScope | undefined): (path: string) => boolean {
  if (!scope || scope.kind === "full") return () => true
  if (scope.kind === "file") return (path) => path === scope.path
  if (scope.kind === "directory") {
    const prefix = scope.path.endsWith("/") ? scope.path : scope.path + "/"
    return (path) => path === scope.path || path.startsWith(prefix)
  }
  if (scope.kind === "files") return (path) => scope.paths.includes(path)
  if (scope.kind === "glob") {
    const matches = createGlobMatcher(scope.pattern)
    return (path) => matches(path)
  }
  if (scope.kind === "delta-kind") return () => false
  return () => true
}

export function createPathSelectorPredicate(selector: string | undefined): (path: string) => boolean {
  if (!selector) return () => true
  return createRestoreScopePredicate(selectorToRestoreScope(selector))
}

export function filterDeltaBySelector(delta: Delta, selector: string | undefined): Delta {
  if (!selector) return delta
  const matches = createPathSelectorPredicate(selector)
  const files = delta.files.filter((file) => matches(file.path))
  return {
    additions: files.filter((file) => file.status === "added").length,
    deletions: files.filter((file) => file.status === "deleted").length,
    modifications: files.filter((file) => file.status === "modified").length,
    renames: files.filter((file) => file.status === "renamed").length,
    files,
  }
}
