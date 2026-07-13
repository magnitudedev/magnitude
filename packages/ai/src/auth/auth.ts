export type AuthApplicator = (headers: Headers) => void

export const Auth = {
  bearer: (token: string): AuthApplicator => (headers) => {
    headers.set("Authorization", `Bearer ${token}`)
  },
  header: (name: string, value: string): AuthApplicator => (headers) => {
    headers.set(name, value)
  },
  none: (() => {}) as AuthApplicator,
}
