import { describe, expect, it } from "vitest"
import { linuxLockHolders } from "./linux-installation-lease"

const procLocks = [
  "1: FLOCK  ADVISORY  READ  4242 08:01:131074 0 EOF",
  "2: FLOCK  ADVISORY  WRITE 777 08:01:999 0 EOF",
  "3: POSIX  ADVISORY  WRITE 555 00:1a:131074 0 EOF",
  "3: -> FLOCK  ADVISORY  WRITE 5151 08:01:131074 0 EOF",
  "",
].join("\n")

describe("Linux installation lock holders", () => {
  it("lists holders and waiters of the lock's inode only", () => {
    expect(linuxLockHolders(procLocks, 131074n)).toEqual([4242, 555, 5151])
    expect(linuxLockHolders(procLocks, 999n)).toEqual([777])
    expect(linuxLockHolders(procLocks, 1n)).toEqual([])
  })
  it("ignores malformed lines", () => {
    expect(linuxLockHolders("garbage\n1: FLOCK\n", 131074n)).toEqual([])
  })
})
