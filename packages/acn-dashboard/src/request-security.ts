export const MUTATING_ACTION_HEADER = 'X-Magnitude-Action'
export const KILL_ALL_ACNS_ACTION = 'kill-all-acns'

/**
 * Destructive dashboard actions require an explicit intent header. Browsers
 * cannot attach this header cross-origin without a successful CORS preflight,
 * and the dashboard API does not opt in to cross-origin requests.
 */
export function isAuthorizedDashboardAction(
  request: Request,
  expectedAction: string,
): boolean {
  if (request.headers.get(MUTATING_ACTION_HEADER) !== expectedAction) {
    return false
  }

  const fetchSite = request.headers.get('Sec-Fetch-Site')
  return fetchSite === null || fetchSite === 'same-origin'
}
