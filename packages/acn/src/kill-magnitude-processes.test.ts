import { describe, expect, it } from "vitest"
import {
  classifyMagnitudeProcess,
  parseProcessList,
} from "../../../scripts/kill-all"

describe("classifyMagnitudeProcess", () => {
  it.each([
    ["/tmp/bin/magnitude-icn serve --port 1234", "ICN"],
    ["target/debug/icn-server serve --fake", "ICN"],
    ["cargo run -p icn-server -- serve --fake", "ICN"],
    ["bun run icn:serve", "ICN"],
    ["/Users/me/.magnitude/bin/magnitude-acn serve --register", "ACN"],
    ["bun run packages/acn/src/binary.ts serve --debug", "ACN"],
    ["/tmp/bin/magnitude-cli --debug", "CLI"],
    ["bun run cli/src/index.tsx --debug", "CLI"],
    ["node packages/cli/bin/magnitude.js", "CLI"],
  ] as const)("classifies %s as %s", (command, expected) => {
    expect(classifyMagnitudeProcess(command)).toBe(expected)
  })

  it.each([
    "bun run scripts/kill-all.ts",
    "bun run packages/acn/src/binary.ts kill-all",
    "code /repo/inference/crates/icn-server/src/main.rs",
    "rg magnitude-cli packages",
    "npm run dev",
  ])("does not classify unrelated command %s", (command) => {
    expect(classifyMagnitudeProcess(command)).toBeUndefined()
  })
})

describe("parseProcessList", () => {
  it("parses ps output while preserving the command", () => {
    expect(parseProcessList("  12     1 /tmp/magnitude-acn serve --register\n  25    12 /tmp/magnitude-icn serve\n")).toEqual([
      { pid: 12, parentPid: 1, command: "/tmp/magnitude-acn serve --register" },
      { pid: 25, parentPid: 12, command: "/tmp/magnitude-icn serve" },
    ])
  })
})
