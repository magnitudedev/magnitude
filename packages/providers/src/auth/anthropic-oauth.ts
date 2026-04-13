/**
 * Anthropic Claude Pro/Max OAuth PKCE flow.
 *
 * Flow:
 * 1. Generate PKCE code verifier + challenge
 * 2. Open browser to claude.ai/oauth/authorize
 * 3. User authenticates, gets code in format: {code}#{state}
 * 4. User pastes code into terminal
 * 5. Exchange code for tokens via console.anthropic.com/v1/oauth/token
 * 6. Store tokens
 */

import crypto from 'crypto'
import type { OAuthAuth } from '../types'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CLIENT_ID = '9d1c250a-e61b-44d9-88ed-5944d1962f5e'
const AUTHORIZE_URL = 'https://claude.ai/oauth/authorize'
const TOKEN_URL = 'https://console.anthropic.com/v1/oauth/token'
const REDIRECT_URI = 'https://console.anthropic.com/oauth/code/callback'
const SCOPES = 'org:create_api_key user:profile user:inference'

/** Beta headers required for OAuth access to Claude 4+ models */
export const ANTHROPIC_OAUTH_BETA_HEADERS = [
  'oauth-2025-04-20',
  'claude-code-20250219',
  'interleaved-thinking-2025-05-14',
]

// ---------------------------------------------------------------------------
// PKCE helpers
// ---------------------------------------------------------------------------

function generateCodeVerifier(): string {
  const buffer = crypto.randomBytes(32)
  return buffer
    .toString('base64')
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=/g, '')
}

function generateCodeChallenge(verifier: string): string {
  const hash = crypto.createHash('sha256').update(verifier).digest()
  return hash
    .toString('base64')
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=/g, '')
}

// ---------------------------------------------------------------------------
// OAuth flow
// ---------------------------------------------------------------------------

export interface AnthropicOAuthStart {
  /** URL to open in the browser */
  authUrl: string
  /** Code verifier for the PKCE exchange — pass back to exchangeAnthropicCode() */
  codeVerifier: string
}

/**
 * Start the Anthropic OAuth PKCE flow.
 * Returns the URL the user should open in their browser and the PKCE verifier.
 */
export function startAnthropicOAuth(): AnthropicOAuthStart {
  const codeVerifier = generateCodeVerifier()
  const codeChallenge = generateCodeChallenge(codeVerifier)

  const authUrl = new URL(AUTHORIZE_URL)
  authUrl.searchParams.set('code', 'true')
  authUrl.searchParams.set('client_id', CLIENT_ID)
  authUrl.searchParams.set('response_type', 'code')
  authUrl.searchParams.set('redirect_uri', REDIRECT_URI)
  authUrl.searchParams.set('scope', SCOPES)
  authUrl.searchParams.set('code_challenge', codeChallenge)
  authUrl.searchParams.set('code_challenge_method', 'S256')
  authUrl.searchParams.set('state', codeVerifier)

  return { authUrl: authUrl.toString(), codeVerifier }
}

/**
 * Exchange the authorization code (pasted by user) for tokens.
 *
 * The code from claude.ai comes in format: {code}#{state}
 */
export async function exchangeAnthropicCode(
  authorizationCode: string,
  codeVerifier: string,
): Promise<OAuthAuth> {
  const parts = authorizationCode.trim().split('#')
  const code = parts[0]
  const state = parts[1]

  const response = await fetch(TOKEN_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      code,
      state,
      grant_type: 'authorization_code',
      client_id: CLIENT_ID,
      redirect_uri: REDIRECT_URI,
      code_verifier: codeVerifier,
    }),
  })

  if (!response.ok) {
    const errorText = await response.text()
    throw new Error(`Anthropic OAuth token exchange failed: ${errorText}`)
  }

  const data: any = await response.json()

  return {
    type: 'oauth',
    oauthMethod: 'oauth-pkce',
    accessToken: data.access_token,
    refreshToken: data.refresh_token,
    expiresAt: Date.now() + data.expires_in * 1000,
  }
}

/**
 * Refresh an expired Anthropic OAuth token.
 */
export async function refreshAnthropicToken(refreshToken: string): Promise<OAuthAuth> {
  const response = await fetch(TOKEN_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      grant_type: 'refresh_token',
      refresh_token: refreshToken,
      client_id: CLIENT_ID,
    }),
  })

  if (!response.ok) {
    const errorText = await response.text()
    throw new Error(`Anthropic OAuth token refresh failed: ${errorText}`)
  }

  const data: any = await response.json()

  return {
    type: 'oauth',
    oauthMethod: 'oauth-pkce',
    accessToken: data.access_token,
    refreshToken: data.refresh_token,
    expiresAt: Date.now() + data.expires_in * 1000,
  }
}
