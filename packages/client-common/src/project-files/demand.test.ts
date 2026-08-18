import { describe, expect, it } from "vitest"
import { Option } from "effect"
import {
  ProjectRelativePathSchema,
  type ProjectDirectoryEntry,
  type ProjectRelativePath,
} from "@magnitudedev/sdk"
import { visitProjectDirectoryDemand } from "./demand"

const path = ProjectRelativePathSchema.make
const directory = (value: string): ProjectDirectoryEntry => ({
  name: value.slice(value.lastIndexOf("/") + 1),
  path: path(value),
  kind: "directory",
  size: Option.none(),
})
const file = (value: string): ProjectDirectoryEntry => ({
  name: value.slice(value.lastIndexOf("/") + 1),
  path: path(value),
  kind: "file",
  size: Option.some(1),
})
const paths = (...values: string[]): ReadonlySet<ProjectRelativePath> =>
  new Set(values.map((value) => path(value)))

describe("visitProjectDirectoryDemand", () => {
  const rootEntries = [directory("packages"), directory("web"), file("README.md")]
  const listings: Readonly<Record<string, readonly ProjectDirectoryEntry[]>> = {
    packages: [directory("packages/acn"), directory("packages/sdk")],
    "packages/acn": [directory("packages/acn/src")],
  }

  it("visits demanded expansion breadth-first through successful ancestors", () => {
    expect(visitProjectDirectoryDemand(
      rootEntries,
      paths("packages", "packages/acn", "packages/acn/src"),
      (directoryPath) => listings[directoryPath],
    )).toEqual([path("packages"), path("packages/acn"), path("packages/acn/src")])
  })

  it("does not visit a deep path before its parent listing exists", () => {
    expect(visitProjectDirectoryDemand(
      rootEntries,
      paths("packages", "packages/acn"),
      () => undefined,
    )).toEqual([path("packages")])
  })

  it("keeps demanded siblings in directory-first traversal order", () => {
    expect(visitProjectDirectoryDemand(
      rootEntries,
      paths("packages", "web"),
      (directoryPath) => listings[directoryPath],
    )).toEqual([path("packages"), path("web")])
  })

  it("ignores demanded paths absent from authoritative parent entries", () => {
    expect(visitProjectDirectoryDemand(
      rootEntries,
      paths("missing", "packages/missing"),
      (directoryPath) => listings[directoryPath],
    )).toEqual([])
  })
})
