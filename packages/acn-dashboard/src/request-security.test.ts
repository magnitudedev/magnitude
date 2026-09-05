import { describe, expect, it } from 'vitest'
import {
  isAuthorizedDashboardAction,
  KILL_ALL_ACNS_ACTION,
  MUTATING_ACTION_HEADER,
} from './request-security'

const request = (headers?: HeadersInit): Request =>
  new Request('http://127.0.0.1:4886/api/acns/kill-all', {
    method: 'POST',
    headers,
  })

describe('isAuthorizedDashboardAction', () => {
  it('rejects a request without an action header', () => {
    expect(isAuthorizedDashboardAction(request(), KILL_ALL_ACNS_ACTION)).toBe(false)
  })

  it('rejects an incorrect action header', () => {
    expect(isAuthorizedDashboardAction(request({
      [MUTATING_ACTION_HEADER]: 'another-action',
    }), KILL_ALL_ACNS_ACTION)).toBe(false)
  })

  it('rejects a cross-site browser request even with the action header', () => {
    expect(isAuthorizedDashboardAction(request({
      [MUTATING_ACTION_HEADER]: KILL_ALL_ACNS_ACTION,
      'Sec-Fetch-Site': 'cross-site',
    }), KILL_ALL_ACNS_ACTION)).toBe(false)
  })

  it('accepts an authorized same-origin dashboard request', () => {
    expect(isAuthorizedDashboardAction(request({
      [MUTATING_ACTION_HEADER]: KILL_ALL_ACNS_ACTION,
      'Sec-Fetch-Site': 'same-origin',
    }), KILL_ALL_ACNS_ACTION)).toBe(true)
  })

  it('accepts an authorized non-browser request without fetch metadata', () => {
    expect(isAuthorizedDashboardAction(request({
      [MUTATING_ACTION_HEADER]: KILL_ALL_ACNS_ACTION,
    }), KILL_ALL_ACNS_ACTION)).toBe(true)
  })
})
