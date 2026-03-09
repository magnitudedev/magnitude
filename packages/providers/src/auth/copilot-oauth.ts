/**
 * GitHub Copilot OAuth device code flow + token exchange.
 *
 * Supports both GitHub.com and GitHub Enterprise.
 *
 * Flow:
 * 1. POST to github.com/login/device/code with client_id + scope
 * 2. Get verification_uri, user_code, device_code, interval
 * 3. Show user: "Visit {url} and enter code: {code}"
 * 4. Poll token URL with device_code
 * 5. Handle authorization_pending (keep polling), slow_down (+5s per RFC 8628)
 * 6. On success, exchange GitHub OAuth token for short-lived Copilot API token
 *
 * Reference: https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps#device-flow
 */

import type { OAuthAuth } from '../types'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CLIENT_ID = 'Iv1.b507a08c87ecfe98'
const SCOPE = 'read:user'
const POLLING_SAFETY_MARGIN_MS = 3000
const COPILOT_TOKEN_URL = 'https://api.github.com/copilot_internal/v2/token'

export const COPILOT_HEADERS: Record<string, string> = {
  'User-Agent': 'GitHubCopilotChat/0.35.0',
  'Editor-Version': 'vscode/1.107.0',
  'Editor-Plugin-Version': 'copilot-chat/0.35.0',
  'Copilot-Integration-Id': 'vscode-chat',
}

function getUrls(domain: string) {
  return {
    deviceCodeUrl: `https://${domain}/login/device/code`,
    accessTokenUrl: `https://${domain}/login/oauth/access_token`,
  }
}

function normalizeDomain(url: string): string {
  return url.replace(/^https?:\/\//, '').replace(/\/$/, '')
}

// ---------------------------------------------------------------------------
// Token exchange
// ---------------------------------------------------------------------------

interface CopilotTokenResponse {
  token: string
  expires_at: number
  refresh_in: number
  endpoints: { api: string }
}

/**
 * Exchange a GitHub OAuth token for a short-lived Copilot API token.
 *
 * The GitHub OAuth token acts as a long-lived refresh token; the returned
 * Copilot token is short-lived (~30 minutes) and must be refreshed periodically.
 */
export async function exchangeCopilotToken(githubOAuthToken: string): Promise<OAuthAuth> {
  const response = await fetch(COPILOT_TOKEN_URL, {
    headers: {
      Accept: 'application/json',
      Authorization: `Bearer ${githubOAuthToken}`,
      ...COPILOT_HEADERS,
    },
  })

  if (!response.ok) {
    const text = await response.text().catch(() => '')
    throw new Error(`Copilot token exchange failed (${response.status}): ${text}`)
  }

  const data: CopilotTokenResponse = await response.json()

  return {
    type: 'oauth',
    accessToken: data.token,                          // Short-lived Copilot API token
    refreshToken: githubOAuthToken,                   // Long-lived GitHub OAuth token
    expiresAt: data.expires_at * 1000 - 5 * 60 * 1000, // 5-minute buffer before actual expiry
  }
}

// ---------------------------------------------------------------------------
// OAuth flow
// ---------------------------------------------------------------------------

export interface CopilotOAuthStart {
  /** URL for the user to visit (e.g., https://github.com/login/device) */
  verificationUrl: string
  /** Code for the user to enter */
  userCode: string
  /** Poll for completion — resolves when user authorizes */
  poll: () => Promise<OAuthAuth>
}

/**
 * Start GitHub Copilot device code OAuth flow.
 *
 * @param enterpriseDomain - Optional GitHub Enterprise domain (e.g., "company.ghe.com").
 *                           Defaults to "github.com".
 */
export async function startCopilotAuth(enterpriseDomain?: string): Promise<CopilotOAuthStart> {
  const domain = enterpriseDomain ? normalizeDomain(enterpriseDomain) : 'github.com'
  const urls = getUrls(domain)

  const deviceResponse = await fetch(urls.deviceCodeUrl, {
    method: 'POST',
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      'User-Agent': 'GitHubCopilotChat/0.35.0',
    },
    body: JSON.stringify({
      client_id: CLIENT_ID,
      scope: SCOPE,
    }),
  })

  if (!deviceResponse.ok) {
    throw new Error('Failed to initiate GitHub Copilot device authorization')
  }

  const deviceData: {
    verification_uri: string
    user_code: string
    device_code: string
    interval: number
  } = await deviceResponse.json()

  const baseInterval = (deviceData.interval || 5) * 1000

  return {
    verificationUrl: deviceData.verification_uri,
    userCode: deviceData.user_code,
    poll: async () => {
      let currentInterval = baseInterval

      while (true) {
        const response = await fetch(urls.accessTokenUrl, {
          method: 'POST',
          headers: {
            Accept: 'application/json',
            'Content-Type': 'application/json',
            'User-Agent': 'GitHubCopilotChat/0.35.0',
          },
          body: JSON.stringify({
            client_id: CLIENT_ID,
            device_code: deviceData.device_code,
            grant_type: 'urn:ietf:params:oauth:grant-type:device_code',
          }),
        })

        if (!response.ok) {
          throw new Error('GitHub Copilot authorization request failed')
        }

        const data: {
          access_token?: string
          error?: string
          interval?: number
        } = await response.json()

        if (data.access_token) {
          // Step 1 complete: got GitHub OAuth token.
          // Step 2: exchange for a short-lived Copilot API token.
          const result = await exchangeCopilotToken(data.access_token)

          if (enterpriseDomain) {
            result.providerSpecific = {
              ...result.providerSpecific,
              enterpriseUrl: domain,
            }
          }

          return result
        }

        if (data.error === 'authorization_pending') {
          await new Promise(r => setTimeout(r, currentInterval + POLLING_SAFETY_MARGIN_MS))
          continue
        }

        if (data.error === 'slow_down') {
          // RFC 8628: add 5 seconds to polling interval
          currentInterval = ((deviceData.interval || 5) + 5) * 1000
          if (data.interval && typeof data.interval === 'number' && data.interval > 0) {
            currentInterval = data.interval * 1000
          }
          await new Promise(r => setTimeout(r, currentInterval + POLLING_SAFETY_MARGIN_MS))
          continue
        }

        if (data.error) {
          throw new Error(`GitHub Copilot authorization failed: ${data.error}`)
        }

        // Unknown state — keep polling
        await new Promise(r => setTimeout(r, currentInterval + POLLING_SAFETY_MARGIN_MS))
      }
    },
  }
}
