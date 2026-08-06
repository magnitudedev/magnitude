import { Schema } from "effect"

export const AcnIdentitySchema = Schema.NonEmptyString.pipe(
  Schema.brand("AcnIdentity"),
)
export type AcnIdentity = typeof AcnIdentitySchema.Type

export const AcnInstanceIdSchema = Schema.NonEmptyString.pipe(
  Schema.brand("AcnInstanceId"),
)
export type AcnInstanceId = typeof AcnInstanceIdSchema.Type

export const ProcessStartIdentitySchema = Schema.NonEmptyString.pipe(
  Schema.brand("ProcessStartIdentity"),
)
export type ProcessStartIdentity = typeof ProcessStartIdentitySchema.Type

export type AcnIdentityComparison = -1 | 0 | 1

interface ParsedSemver {
  readonly major: string
  readonly minor: string
  readonly patch: string
  readonly prerelease: ReadonlyArray<string>
  readonly build: string | null
}

const semver = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)(?:\.(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$/

const parse = (value: string): ParsedSemver | null => {
  const match = semver.exec(value)
  if (match === null) return null
  return {
    major: match[1]!,
    minor: match[2]!,
    patch: match[3]!,
    prerelease: match[4]?.split(".") ?? [],
    build: match[5] ?? null,
  }
}

const compareIdentifiers = (left: string, right: string): AcnIdentityComparison => {
  const leftNumeric = /^\d+$/.test(left)
  const rightNumeric = /^\d+$/.test(right)
  if (leftNumeric && rightNumeric) {
    if (left.length !== right.length) return left.length < right.length ? -1 : 1
    return left < right ? -1 : left > right ? 1 : 0
  }
  if (leftNumeric !== rightNumeric) return leftNumeric ? -1 : 1
  return left < right ? -1 : left > right ? 1 : 0
}

const naturalCompare = (left: string, right: string): AcnIdentityComparison => {
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
      const normalizedLeft = leftPart.replace(/^0+(?=\d)/, "")
      const normalizedRight = rightPart.replace(/^0+(?=\d)/, "")
      if (normalizedLeft.length !== normalizedRight.length) {
        return normalizedLeft.length < normalizedRight.length ? -1 : 1
      }
      if (normalizedLeft !== normalizedRight) return normalizedLeft < normalizedRight ? -1 : 1
      continue
    }
    return leftPart < rightPart ? -1 : 1
  }
  return left < right ? -1 : 1
}

const developmentTimestamp = (build: string): string | null =>
  /^dev\.[0-9A-Za-z-]+\.(0|[1-9]\d*)$/.exec(build)?.[1] ?? null

/** Total ordering for ACN release identities, including development builds. */
export const compareAcnIdentities = (
  candidate: AcnIdentity | string,
  incumbent: AcnIdentity | string,
): AcnIdentityComparison => {
  if (candidate === incumbent) return 0
  const left = parse(candidate)
  const right = parse(incumbent)
  if (left === null || right === null) return naturalCompare(candidate, incumbent)

  for (const field of ["major", "minor", "patch"] as const) {
    const comparison = compareIdentifiers(left[field], right[field])
    if (comparison !== 0) return comparison
  }

  if (left.prerelease.length === 0 || right.prerelease.length === 0) {
    if (left.prerelease.length !== right.prerelease.length) {
      return left.prerelease.length === 0 ? 1 : -1
    }
  }
  const prereleaseLength = Math.max(left.prerelease.length, right.prerelease.length)
  for (let index = 0; index < prereleaseLength; index += 1) {
    const leftIdentifier = left.prerelease[index]
    const rightIdentifier = right.prerelease[index]
    if (leftIdentifier === undefined) return -1
    if (rightIdentifier === undefined) return 1
    const comparison = compareIdentifiers(leftIdentifier, rightIdentifier)
    if (comparison !== 0) return comparison
  }

  const leftDevelopment = left.build === null ? null : developmentTimestamp(left.build)
  const rightDevelopment = right.build === null ? null : developmentTimestamp(right.build)
  if (leftDevelopment !== null || rightDevelopment !== null) {
    if (leftDevelopment === null) return -1
    if (rightDevelopment === null) return 1
  }
  if (left.build === null) return 1
  if (right.build === null) return -1
  if (leftDevelopment !== null && rightDevelopment !== null) {
    const comparison = compareIdentifiers(leftDevelopment, rightDevelopment)
    if (comparison !== 0) return comparison
  }
  return naturalCompare(left.build, right.build)
}

export const canUseAcnIdentity = (expected: AcnIdentity | string, observed: AcnIdentity | string): boolean =>
  compareAcnIdentities(expected, observed) <= 0
