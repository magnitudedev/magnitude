type Comparison = -1 | 0 | 1

interface ParsedSemver {
  readonly major: string
  readonly minor: string
  readonly patch: string
  readonly prerelease: ReadonlyArray<string>
  readonly build: string | null
}

const SEMVER = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)(?:\.(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$/

const parse = (value: string): ParsedSemver | null => {
  const match = SEMVER.exec(value)
  if (!match) return null
  return {
    major: match[1]!,
    minor: match[2]!,
    patch: match[3]!,
    prerelease: match[4]?.split(".") ?? [],
    build: match[5] ?? null,
  }
}

const compareIdentifiers = (left: string, right: string): Comparison => {
  const leftNumeric = /^\d+$/.test(left)
  const rightNumeric = /^\d+$/.test(right)
  if (leftNumeric && rightNumeric) {
    if (left.length !== right.length) return left.length < right.length ? -1 : 1
    return left < right ? -1 : left > right ? 1 : 0
  }
  if (leftNumeric !== rightNumeric) return leftNumeric ? -1 : 1
  return left < right ? -1 : left > right ? 1 : 0
}

const naturalCompare = (left: string, right: string): Comparison => {
  if (left === right) return 0
  const leftParts = left.match(/\d+|\D+/g) ?? []
  const rightParts = right.match(/\d+|\D+/g) ?? []
  const length = Math.max(leftParts.length, rightParts.length)

  for (let index = 0; index < length; index += 1) {
    const leftPart = leftParts[index]
    const rightPart = rightParts[index]
    if (leftPart === undefined) return -1
    if (rightPart === undefined) return 1
    if (leftPart === rightPart) continue

    const leftNumeric = /^\d+$/.test(leftPart)
    const rightNumeric = /^\d+$/.test(rightPart)
    if (leftNumeric && rightNumeric) {
      const leftNumber = leftPart.replace(/^0+(?=\d)/, "")
      const rightNumber = rightPart.replace(/^0+(?=\d)/, "")
      if (leftNumber.length !== rightNumber.length) {
        return leftNumber.length < rightNumber.length ? -1 : 1
      }
      if (leftNumber !== rightNumber) return leftNumber < rightNumber ? -1 : 1
      continue
    }

    return leftPart < rightPart ? -1 : 1
  }

  // Different strings whose natural components have equal values (for
  // example, leading-zero variants) still receive a stable total order.
  return left < right ? -1 : 1
}

const devTimestamp = (build: string): string | null =>
  /^dev\.[0-9A-Za-z-]+\.(0|[1-9]\d*)$/.exec(build)?.[1] ?? null

/**
 * Magnitude release ordering.
 *
 * Published versions use SemVer precedence. Development identities use the
 * generated `+dev.<commit>.<timestamp>` suffix. Equal-precedence build
 * identities are naturally ordered, so numeric segments such as timestamps
 * compare numerically. A published release outranks a build of the same base.
 * Arbitrary non-SemVer strings use the same deterministic natural ordering.
 */
export const compareReleaseVersions = (
  candidate: string,
  incumbent: string,
): Comparison => {
  if (candidate === incumbent) return 0
  const left = parse(candidate)
  const right = parse(incumbent)
  if (!left || !right) return naturalCompare(candidate, incumbent)

  for (const field of ["major", "minor", "patch"] as const) {
    const compared = compareIdentifiers(left[field], right[field])
    if (compared !== 0) return compared
  }

  if (left.prerelease.length === 0 || right.prerelease.length === 0) {
    if (left.prerelease.length !== right.prerelease.length) {
      return left.prerelease.length === 0 ? 1 : -1
    }
  }
  const length = Math.max(left.prerelease.length, right.prerelease.length)
  for (let index = 0; index < length; index += 1) {
    const leftIdentifier = left.prerelease[index]
    const rightIdentifier = right.prerelease[index]
    if (leftIdentifier === undefined) return -1
    if (rightIdentifier === undefined) return 1
    const compared = compareIdentifiers(leftIdentifier, rightIdentifier)
    if (compared !== 0) return compared
  }

  if (left.build === null) return 1
  if (right.build === null) return -1
  const leftDevTimestamp = devTimestamp(left.build)
  const rightDevTimestamp = devTimestamp(right.build)
  if (leftDevTimestamp !== null && rightDevTimestamp !== null) {
    const compared = compareIdentifiers(leftDevTimestamp, rightDevTimestamp)
    if (compared !== 0) return compared
  }
  return naturalCompare(left.build, right.build)
}
