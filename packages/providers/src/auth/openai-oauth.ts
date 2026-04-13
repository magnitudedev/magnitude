/**
 * OpenAI ChatGPT Pro/Plus OAuth flows.
 *
 * Supports two methods:
 * 1. Browser PKCE — local server on port 1455 receives callback
 * 2. Headless Device Code — user enters code on auth.openai.com/codex/device
 *
 * Reference: https://developers.openai.com/codex/auth/
 */

import type { OAuthAuth } from '../types'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CLIENT_ID = 'app_EMoamEEZ73f0CkXaXp7hrann'
const ISSUER = 'https://auth.openai.com'
const OAUTH_PORT = 1455
const POLLING_SAFETY_MARGIN_MS = 3000

// ---------------------------------------------------------------------------
// PKCE helpers
// ---------------------------------------------------------------------------

function generateRandomString(length: number): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~'
  const bytes = crypto.getRandomValues(new Uint8Array(length))
  return Array.from(bytes)
    .map((b) => chars[b % chars.length])
    .join('')
}

function base64UrlEncode(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer)
  const binary = String.fromCharCode(...bytes)
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
}

async function generatePKCE(): Promise<{ verifier: string; challenge: string }> {
  const verifier = generateRandomString(43)
  const encoder = new TextEncoder()
  const data = encoder.encode(verifier)
  const hash = await crypto.subtle.digest('SHA-256', data)
  const challenge = base64UrlEncode(hash)
  return { verifier, challenge }
}

function generateState(): string {
  return base64UrlEncode(crypto.getRandomValues(new Uint8Array(32)).buffer as ArrayBuffer)
}

// ---------------------------------------------------------------------------
// JWT parsing for account ID extraction
// ---------------------------------------------------------------------------

interface IdTokenClaims {
  chatgpt_account_id?: string
  organizations?: Array<{ id: string }>
  'https://api.openai.com/auth'?: { chatgpt_account_id?: string }
}

function parseJwtClaims(token: string): IdTokenClaims | undefined {
  const parts = token.split('.')
  if (parts.length !== 3) return undefined
  try {
    return JSON.parse(Buffer.from(parts[1], 'base64url').toString())
  } catch {
    return undefined
  }
}

function extractAccountId(tokens: TokenResponse): string | undefined {
  if (tokens.id_token) {
    const claims = parseJwtClaims(tokens.id_token)
    if (claims) {
      const id =
        claims.chatgpt_account_id ||
        claims['https://api.openai.com/auth']?.chatgpt_account_id ||
        claims.organizations?.[0]?.id
      if (id) return id
    }
  }
  if (tokens.access_token) {
    const claims = parseJwtClaims(tokens.access_token)
    if (claims) {
      return (
        claims.chatgpt_account_id ||
        claims['https://api.openai.com/auth']?.chatgpt_account_id ||
        claims.organizations?.[0]?.id
      )
    }
  }
  return undefined
}

interface TokenResponse {
  id_token: string
  access_token: string
  refresh_token: string
  expires_in?: number
}

// ---------------------------------------------------------------------------
// Method 1: Browser PKCE (local callback server on port 1455)
// ---------------------------------------------------------------------------

export interface OpenAIBrowserOAuthStart {
  /** URL to open in the browser */
  authUrl: string
  /** Promise that resolves when the user completes OAuth in the browser */
  waitForCallback: () => Promise<OAuthAuth>
  /** Stop the local callback server */
  stop: () => void
}

const HTML_SUCCESS = `<!doctype html><html><head><title>Magnitude - Authorization Successful</title>
<style>body{font-family:system-ui;display:flex;justify-content:center;align-items:center;height:100vh;margin:0;background:#131010;color:#f1ecec}.c{text-align:center;padding:2rem}p{color:#b7b1b1}</style>
</head><body><div class="c"><h1>Authorization Successful</h1><p>You can close this window and return to Magnitude.</p></div>
<script>setTimeout(()=>window.close(),2000)</script></body></html>`

const HTML_ERROR = (error: string) => `<!doctype html><html><head><title>Magnitude - Authorization Failed</title>
<style>body{font-family:system-ui;display:flex;justify-content:center;align-items:center;height:100vh;margin:0;background:#131010;color:#f1ecec}.c{text-align:center;padding:2rem}p{color:#b7b1b1}.e{color:#ff917b;font-family:monospace;margin-top:1rem;padding:1rem;background:#3c140d;border-radius:0.5rem}</style>
</head><body><div class="c"><h1>Authorization Failed</h1><p>An error occurred during authorization.</p><div class="e">${error}</div></div></body></html>`

/**
 * Start OpenAI browser-based PKCE OAuth.
 * Spins up a local HTTP server on port 1455 and waits for the callback.
 */
export async function startOpenAIBrowserOAuth(): Promise<OpenAIBrowserOAuthStart> {
  const pkce = await generatePKCE()
  const state = generateState()
  const redirectUri = `http://localhost:${OAUTH_PORT}/auth/callback`

  let resolveCallback: ((auth: OAuthAuth) => void) | null = null
  let rejectCallback: ((err: Error) => void) | null = null

  const callbackPromise = new Promise<OAuthAuth>((resolve, reject) => {
    resolveCallback = resolve
    rejectCallback = reject
    // 5 minute timeout
    setTimeout(() => reject(new Error('OAuth callback timeout')), 5 * 60 * 1000)
  })

  const server = Bun.serve({
    port: OAUTH_PORT,
    fetch(req) {
      const url = new URL(req.url)
      if (url.pathname === '/auth/callback') {
        const code = url.searchParams.get('code')
        const error = url.searchParams.get('error')
        const errorDesc = url.searchParams.get('error_description')

        if (error) {
          rejectCallback?.(new Error(errorDesc || error))
          return new Response(HTML_ERROR(errorDesc || error), { headers: { 'Content-Type': 'text/html' } })
        }

        if (!code) {
          rejectCallback?.(new Error('Missing authorization code'))
          return new Response(HTML_ERROR('Missing authorization code'), { status: 400, headers: { 'Content-Type': 'text/html' } })
        }

        // Exchange code for tokens in background
        exchangeOpenAICode(code, redirectUri, pkce.verifier)
          .then((auth) => resolveCallback?.(auth))
          .catch((err) => rejectCallback?.(err))

        return new Response(HTML_SUCCESS, { headers: { 'Content-Type': 'text/html' } })
      }
      return new Response('Not found', { status: 404 })
    },
  })

  const params = new URLSearchParams({
    response_type: 'code',
    client_id: CLIENT_ID,
    redirect_uri: redirectUri,
    scope: 'openid profile email offline_access',
    code_challenge: pkce.challenge,
    code_challenge_method: 'S256',
    id_token_add_organizations: 'true',
    codex_cli_simplified_flow: 'true',
    state,
    originator: 'magnitude',
  })

  const authUrl = `${ISSUER}/oauth/authorize?${params.toString()}`

  return {
    authUrl,
    waitForCallback: async () => {
      try {
        return await callbackPromise
      } finally {
        server.stop()
      }
    },
    stop: () => server.stop(),
  }
}

async function exchangeOpenAICode(code: string, redirectUri: string, verifier: string): Promise<OAuthAuth> {
  const response = await fetch(`${ISSUER}/oauth/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      grant_type: 'authorization_code',
      code,
      redirect_uri: redirectUri,
      client_id: CLIENT_ID,
      code_verifier: verifier,
    }).toString(),
  })

  if (!response.ok) {
    throw new Error(`OpenAI token exchange failed: ${response.status}`)
  }

  const tokens: TokenResponse = await response.json()
  const accountId = extractAccountId(tokens)

  return {
    type: 'oauth',
    oauthMethod: 'oauth-browser',
    accessToken: tokens.access_token,
    refreshToken: tokens.refresh_token,
    expiresAt: Date.now() + (tokens.expires_in ?? 3600) * 1000,
    accountId,
  }
}

// ---------------------------------------------------------------------------
// Method 2: Headless Device Code
// ---------------------------------------------------------------------------

export interface OpenAIDeviceOAuthStart {
  /** URL for the user to visit */
  verificationUrl: string
  /** Code for the user to enter */
  userCode: string
  /** Poll for completion — resolves when user authorizes */
  poll: () => Promise<OAuthAuth>
}

/**
 * Start OpenAI headless device code flow.
 * Returns a URL and code for the user to enter, plus a poll function.
 */
export async function startOpenAIDeviceOAuth(): Promise<OpenAIDeviceOAuthStart> {
  const deviceResponse = await fetch(`${ISSUER}/api/accounts/deviceauth/usercode`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'User-Agent': 'magnitude-cli',
    },
    body: JSON.stringify({ client_id: CLIENT_ID }),
  })

  if (!deviceResponse.ok) {
    throw new Error('Failed to initiate OpenAI device authorization')
  }

  const deviceData: any = await deviceResponse.json()
  const interval = Math.max(parseInt(deviceData.interval) || 5, 1) * 1000

  return {
    verificationUrl: `${ISSUER}/codex/device`,
    userCode: deviceData.user_code,
    poll: async () => {
      while (true) {
        const response = await fetch(`${ISSUER}/api/accounts/deviceauth/token`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'User-Agent': 'magnitude-cli',
          },
          body: JSON.stringify({
            device_auth_id: deviceData.device_auth_id,
            user_code: deviceData.user_code,
          }),
        })

        if (response.ok) {
          const data: any = await response.json()
          // Exchange the authorization code for real tokens
          const tokenResponse = await fetch(`${ISSUER}/oauth/token`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: new URLSearchParams({
              grant_type: 'authorization_code',
              code: data.authorization_code,
              redirect_uri: `${ISSUER}/deviceauth/callback`,
              client_id: CLIENT_ID,
              code_verifier: data.code_verifier,
            }).toString(),
          })

          if (!tokenResponse.ok) {
            throw new Error(`OpenAI device token exchange failed: ${tokenResponse.status}`)
          }

          const tokens: TokenResponse = await tokenResponse.json()
          const accountId = extractAccountId(tokens)

          return {
            type: 'oauth' as const,
            oauthMethod: 'oauth-device' as const,
            accessToken: tokens.access_token,
            refreshToken: tokens.refresh_token,
            expiresAt: Date.now() + (tokens.expires_in ?? 3600) * 1000,
            accountId,
          }
        }

        // 403/404 = authorization pending, keep polling
        if (response.status !== 403 && response.status !== 404) {
          throw new Error('OpenAI device authorization failed')
        }

        await new Promise(r => setTimeout(r, interval + POLLING_SAFETY_MARGIN_MS))
      }
    },
  }
}

/**
 * Refresh an expired OpenAI OAuth token.
 */
export async function refreshOpenAIToken(
  refreshToken: string,
  oauthMethod: 'oauth-browser' | 'oauth-device' = 'oauth-browser',
): Promise<OAuthAuth> {
  const response = await fetch(`${ISSUER}/oauth/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      grant_type: 'refresh_token',
      refresh_token: refreshToken,
      client_id: CLIENT_ID,
    }).toString(),
  })

  if (!response.ok) {
    throw new Error(`OpenAI token refresh failed: ${response.status}`)
  }

  const tokens: TokenResponse = await response.json()
  const accountId = extractAccountId(tokens)

  return {
    type: 'oauth',
    oauthMethod,
    accessToken: tokens.access_token,
    refreshToken: tokens.refresh_token,
    expiresAt: Date.now() + (tokens.expires_in ?? 3600) * 1000,
    accountId,
  }
}
