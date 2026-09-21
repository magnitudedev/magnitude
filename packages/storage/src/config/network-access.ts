import { randomBytes } from 'node:crypto'
import { isIP } from 'node:net'
import { Option } from 'effect'
import type { NetworkAccessConfig } from '../types/config'

export const LOOPBACK_BIND = '127.0.0.1'
export const ALL_INTERFACES_BIND = '0.0.0.0'

export interface NetworkAccess {
  readonly enabled: boolean
  readonly bind: string
  readonly apiKey: Option.Option<string>
  readonly requireApiKey: boolean
  readonly allowedHosts: ReadonlyArray<string>
  readonly warning: Option.Option<string>
}

export const LOOPBACK_ONLY: NetworkAccess = {
  enabled: false,
  bind: LOOPBACK_BIND,
  apiKey: Option.none(),
  requireApiKey: true,
  allowedHosts: [],
  warning: Option.none(),
}

/** Turns the saved `network` object into what the service binds and enforces. Disabled means loopback only. */
export const resolveNetworkAccess = (config: Option.Option<NetworkAccessConfig>): NetworkAccess => {
  if (Option.isNone(config) || !config.value.enabled) return LOOPBACK_ONLY
  const value = config.value
  const bind = value.bind?.trim()
  const bindIsAddress = bind !== undefined && isIP(bind) !== 0
  return {
    enabled: true,
    bind: bindIsAddress ? bind : ALL_INTERFACES_BIND,
    apiKey: Option.fromNullable(value.apiKey),
    requireApiKey: value.requireApiKey,
    allowedHosts: value.allowedHosts,
    warning: bind !== undefined && !bindIsAddress
      ? Option.some(`Ignoring network.bind "${bind}": it must be an IP address. Listening on all interfaces.`)
      : Option.none(),
  }
}

export const generateApiKey = (): string => `mag-${randomBytes(24).toString('base64url')}`

const LOOPBACK_ADDRESSES = new Set(['127.0.0.1', '::1', '::ffff:127.0.0.1'])

/** Whether a socket peer address is this machine's loopback. */
export const isLoopbackAddress = (address: string): boolean => {
  const bare = address.replace(/^\[|\]$/g, '')
  return LOOPBACK_ADDRESSES.has(bare) || bare.startsWith('127.')
}

const LOCAL_HOST_NAMES = new Set(['localhost', '127.0.0.1', '[::1]'])
const BUILT_IN_ALLOWED_HOSTS = ['host.docker.internal']

const splitHostPort = (header: string): string => {
  const trimmed = header.trim().toLowerCase()
  if (trimmed.startsWith('[')) return trimmed.replace(/\]:\d+$/, ']')
  return trimmed.replace(/:\d+$/, '')
}

/**
 * The Host header a request may carry. Local names are always accepted. With network access on,
 * IP literals, Docker's host name, Tailscale MagicDNS names, and configured names are accepted as
 * well. Anything else is refused, which is what defeats DNS rebinding from a browser.
 */
export const isAllowedHostHeader = (header: string | undefined, network: NetworkAccess): boolean => {
  if (header === undefined) return false
  const host = splitHostPort(header)
  if (LOCAL_HOST_NAMES.has(host)) return true
  if (!network.enabled) return false
  if (isIP(host.replace(/^\[|\]$/g, '')) !== 0) return true
  if (BUILT_IN_ALLOWED_HOSTS.includes(host)) return true
  if (host.endsWith('.ts.net')) return true
  return network.allowedHosts.some(allowed => allowed.trim().toLowerCase() === host)
}

/** Whether a remote inference request carries the configured key, when one is required. */
export const authorizesRemoteInference = (
  headers: { readonly authorization?: string; readonly 'x-api-key'?: string },
  network: NetworkAccess,
): boolean => {
  if (!network.requireApiKey) return true
  if (Option.isNone(network.apiKey)) return false
  const bearer = headers.authorization?.match(/^Bearer\s+(.+)$/i)?.[1]?.trim()
  const presented = bearer ?? headers['x-api-key']?.trim()
  return presented !== undefined && constantTimeEqual(presented, network.apiKey.value)
}

const constantTimeEqual = (a: string, b: string): boolean => {
  const left = Buffer.from(a)
  const right = Buffer.from(b)
  if (left.length !== right.length) return false
  let difference = 0
  for (let index = 0; index < left.length; index += 1) difference |= left[index]! ^ right[index]!
  return difference === 0
}
