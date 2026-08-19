import { describe, expect, it } from "vitest"
import { Schema } from "effect"
import {
  MAX_TEXT_FILE_UPLOAD_BYTES,
  RawMessageUpload,
  RawMessageUploads,
} from "./attachments"

describe("RawMessageUpload", () => {
  it("decodes text-file snapshots through the message upload union", () => {
    expect(Schema.decodeUnknownSync(RawMessageUpload)({
      type: "raw_text_file",
      filename: "notes.md",
      data: Buffer.from("# Notes\n").toString("base64"),
    })).toEqual({
      type: "raw_text_file",
      filename: "notes.md",
      data: "IyBOb3Rlcwo=",
    })
  })

  it("rejects payload strings that cannot represent the bounded text size", () => {
    const maximumBase64Length = Math.ceil(MAX_TEXT_FILE_UPLOAD_BYTES / 3) * 4
    expect(() => Schema.decodeUnknownSync(RawMessageUpload)({
      type: "raw_text_file",
      filename: "large.txt",
      data: "A".repeat(maximumBase64Length + 1),
    })).toThrow()
  })

  it("rejects more than 20 uploads at the wire boundary", () => {
    expect(() => Schema.decodeUnknownSync(RawMessageUploads)(
      Array.from({ length: 21 }, (_, index) => ({
        type: "raw_text_file",
        filename: `${index}.txt`,
        data: "YQ==",
      })),
    )).toThrow()
  })
})
