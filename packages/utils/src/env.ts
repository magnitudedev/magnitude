/**
 * Parse a string env var as a boolean flag.
 * Treats `"0"`, `"false"`, `"no"`, `"off"`, empty, and unset as false.
 * Anything else (including `"1"`, `"true"`) is true.
 */
export function isEnvFlagOn(value: string | undefined): boolean {
  if (value === undefined) return false
  const v = value.trim().toLowerCase()
  return v !== '' && v !== '0' && v !== 'false' && v !== 'no' && v !== 'off'
}
