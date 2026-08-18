import { describe, expect, it } from "vitest"
import { Schema } from "effect"
import { ProjectEntryMoveSchema, ProjectFilesChangeSchema, ProjectRelativePathSchema } from "./project-files"

const decode = Schema.decodeUnknownEither(ProjectRelativePathSchema)

describe("ProjectRelativePathSchema", () => {
  it.each(["", "src/index.ts", ".github/workflows/ci.yml"])("accepts %j", (path) => {
    expect(decode(path)._tag).toBe("Right")
  })

  it.each(["/etc/passwd", "../secret", "src/../secret", "./src", "src//file", "C:/secret", "src\\file", "a\0b"])("rejects %j", (path) => {
    expect(decode(path)._tag).toBe("Left")
  })
})

describe("ProjectEntryMoveSchema", () => {
  it("preserves the exact authoritative move acknowledgement", () => {
    const decoded = Schema.decodeUnknownEither(ProjectEntryMoveSchema)({
      sourcePath: "src",
      destinationPath: "packages/src",
      kind: "directory",
    })
    expect(decoded._tag).toBe("Right")
  })

  it("rejects invalid paths and entry kinds", () => {
    expect(Schema.decodeUnknownEither(ProjectEntryMoveSchema)({
      sourcePath: "../src",
      destinationPath: "packages/src",
      kind: "directory",
    })._tag).toBe("Left")
    expect(Schema.decodeUnknownEither(ProjectEntryMoveSchema)({
      sourcePath: "src",
      destinationPath: "packages/src",
      kind: "symlink",
    })._tag).toBe("Left")
  })
})

describe("ProjectFilesChangeSchema", () => {
  it("contains only project identity", () => {
    expect(Schema.decodeUnknownEither(ProjectFilesChangeSchema)({
      projectId: "project-1",
    })._tag).toBe("Right")
    expect(Schema.decodeUnknownEither(ProjectFilesChangeSchema)({
      projectId: "",
    })._tag).toBe("Left")
  })
})
