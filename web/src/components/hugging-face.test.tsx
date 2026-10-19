import { renderToStaticMarkup } from "react-dom/server"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { Exit, Option } from "effect"
import { Result } from "@effect-atom/atom-react"
import { HuggingFaceAccessRequired, type HuggingFaceAccount, type HuggingFaceDeviceSignIn, type HuggingFaceSignInAttemptId } from "@magnitudedev/sdk"
import { HuggingFaceAccessActions, HuggingFaceRow, HuggingFaceRowView, type DeviceCode } from "./hugging-face"
import { candidateErrorNotice } from "./candidate-errors"
import { ErrorNotice } from "./error-notice"
import { downloadNotice } from "./model-error"

const fake = vi.hoisted(() => {
  const atoms = { account: { atom: "account" }, start: { atom: "start" }, startDevice: { atom: "startDevice" }, cancel: { atom: "cancel" }, signOut: { atom: "signOut" } }
  return {
    atoms,
    account: { current: undefined as unknown },
    client: { HuggingFace: { GetHuggingFaceAccount: () => atoms.account, StartHuggingFaceSignIn: atoms.start, StartHuggingFaceDeviceSignIn: atoms.startDevice, CancelHuggingFaceSignIn: atoms.cancel, SignOutOfHuggingFace: atoms.signOut } },
    setters: new Map<unknown, ReturnType<typeof vi.fn>>(),
    values: new Map<unknown, unknown>(),
    session: { clientWindow: { _tag: "None" } as unknown },
  }
})

vi.mock("@/session", () => ({ useSession: () => fake.session }))

vi.mock("@magnitudedev/client-common", async importOriginal => ({
  ...await importOriginal<typeof import("@magnitudedev/client-common")>(),
  useAgentClient: () => fake.client,
}))

vi.mock("@effect-atom/atom-react", async importOriginal => {
  const original = await importOriginal<typeof import("@effect-atom/atom-react")>()
  return {
    ...original,
    useAtomValue: (atom: unknown) => atom === fake.atoms.account ? { result: fake.account.current } : fake.values.get(atom) ?? original.Result.initial(),
    useAtomSet: (atom: unknown) => {
      const setter = fake.setters.get(atom) ?? vi.fn()
      fake.setters.set(atom, setter)
      return setter
    },
  }
})

const signedOut: HuggingFaceAccount = { _tag: "SignedOut", reason: Option.none() }
const view = (account: Result.Result<HuggingFaceAccount, unknown>, notice?: { title: string; description?: string }, deviceCode: Option.Option<DeviceCode> = Option.none()) =>
  renderToStaticMarkup(<HuggingFaceRowView account={account} pending={false} notice={notice} deviceCode={deviceCode} onSignIn={() => {}} onCancel={() => {}} onSignOut={() => {}}
    onCopyCode={() => Promise.resolve(true)} onOpenVerification={() => {}} />)
const signingInWithCode: HuggingFaceAccount = { _tag: "SigningIn", method: "Device" }
const deviceStart = (verificationUriComplete: Option.Option<string>): HuggingFaceDeviceSignIn => ({
  attemptId: "attempt-1" as HuggingFaceSignInAttemptId,
  verificationUri: "https://hf.co/oauth/device",
  verificationUriComplete,
  userCode: "WXYZ-1234",
  expiresIn: 300,
  interval: 5,
})
type RowProps = { onSignIn: () => void; onCancel: () => void; onSignOut: () => void; onCopyCode: () => Promise<boolean>; onOpenVerification: () => void }
const desktopWindow = () => { fake.session.clientWindow = Option.some({ platform: "darwin" }) }
const button = (html: string, label: string) => html.match(new RegExp(`<button[^>]*>${label}</button>`))

afterEach(() => {
  fake.setters.clear()
  fake.values.clear()
  fake.session.clientWindow = Option.none()
  vi.unstubAllGlobals()
})

describe("Hugging Face settings row", () => {
  it("offers Sign in when signed out, with why the last sign-in ended", () => {
    const html = view(Result.success(signedOut))
    expect(html).toContain("Hugging Face")
    expect(html).toContain("Not signed in. Sign in for gated models, private repositories and higher download limits.")
    expect(button(html, "Sign in")).not.toBeNull()
    const expired = view(Result.success({ _tag: "SignedOut", reason: Option.some("SessionExpired") }))
    expect(expired).toContain("Your Hugging Face sign-in expired. Sign in again.")
    expect(view(Result.success(signedOut), { title: "Couldn’t start signing in to Hugging Face", description: "Port in use" })).toMatch(/Couldn’t start signing in to Hugging Face.*Port in use/)
  })
  it("waits for the browser with Cancel, and for the code when the CLI started it", () => {
    const browser = view(Result.success({ _tag: "SigningIn", method: "Browser" }))
    expect(browser).toContain("Waiting for the browser…")
    expect(button(browser, "Cancel")).not.toBeNull()
    expect(button(browser, "Sign in")).toBeNull()
    const device = view(Result.success({ _tag: "SigningIn", method: "Device" }))
    expect(device).toContain("Waiting for the code to be entered…")
    expect(button(device, "Cancel")).not.toBeNull()
  })
  it("names the signed-in account with its avatar and offers Sign out", () => {
    const html = view(Result.success({ _tag: "SignedIn", username: "trg", avatarUrl: Option.some("https://huggingface.co/avatars/trg.png") }))
    expect(html).toContain("Signed in as trg")
    expect(html).toContain('src="https://huggingface.co/avatars/trg.png"')
    expect(button(html, "Sign out")).not.toBeNull()
    expect(view(Result.success({ _tag: "SignedIn", username: "trg", avatarUrl: Option.none() }))).not.toContain("avatars/")
  })
  it("says when the account can't be read", () => {
    expect(view(Result.fail(new Error("transport")))).toContain("Couldn’t read your Hugging Face account")
  })
  it("starts a browser sign-in in the desktop app and opens its page, cancels, and signs out", async () => {
    desktopWindow()
    const open = vi.fn()
    vi.stubGlobal("window", { open })
    fake.setters.set(fake.atoms.start, vi.fn(() => Promise.resolve(Exit.succeed({ authorizationUrl: "https://huggingface.co/oauth/authorize?state=x" }))))
    fake.account.current = Result.success(signedOut)
    const row = HuggingFaceRow() as ReactElement<RowProps>
    row.props.onSignIn()
    await vi.waitFor(() => expect(open).toHaveBeenCalledWith("https://huggingface.co/oauth/authorize?state=x", "_blank", "noopener,noreferrer"))
    expect(fake.setters.get(fake.atoms.start)).toHaveBeenCalledWith({})
    expect(fake.setters.get(fake.atoms.startDevice)).not.toHaveBeenCalled()
    row.props.onCancel()
    expect(fake.setters.get(fake.atoms.cancel)).toHaveBeenCalledWith({ attemptId: Option.none() })
    row.props.onSignOut()
    expect(fake.setters.get(fake.atoms.signOut)).toHaveBeenCalledWith({})
  })
  it("opens nothing when the sign-in can't start", async () => {
    desktopWindow()
    const open = vi.fn()
    vi.stubGlobal("window", { open })
    const start = vi.fn(() => Promise.resolve(Exit.fail("failed")))
    fake.setters.set(fake.atoms.start, start)
    fake.account.current = Result.success(signedOut)
    const row = HuggingFaceRow() as ReactElement<{ onSignIn: () => void }>
    row.props.onSignIn()
    await vi.waitFor(() => expect(start).toHaveBeenCalled())
    await Promise.resolve()
    expect(open).not.toHaveBeenCalled()
  })
})

describe("Hugging Face device sign-in in a browser", () => {
  const withCode = (complete: Option.Option<string> = Option.none(), timestamp = Date.now()) =>
    fake.values.set(fake.atoms.startDevice, Result.success(deviceStart(complete), { timestamp }))
  it("starts a device sign-in without a desktop host and shows the code with Copy and the link", () => {
    const open = vi.fn()
    vi.stubGlobal("window", { open })
    fake.account.current = Result.success(signedOut)
    ;(HuggingFaceRow() as ReactElement<RowProps>).props.onSignIn()
    expect(fake.setters.get(fake.atoms.startDevice)).toHaveBeenCalledWith({})
    expect(fake.setters.get(fake.atoms.start)).not.toHaveBeenCalled()
    expect(open).not.toHaveBeenCalled()
    withCode()
    fake.account.current = Result.success(signingInWithCode)
    const html = renderToStaticMarkup(HuggingFaceRow())
    expect(html).toContain("Enter this code on Hugging Face")
    expect(html).toContain("WXYZ-1234")
    expect(html).toMatch(/<button[^>]*>.*Copy<\/button>/)
    expect(html).toMatch(/<button[^>]*>Open huggingface.co.*<\/button>/)
    expect(html).toContain("The code expires at")
    expect(html).toContain("Waiting for the code to be entered…")
    expect(button(html, "Cancel")).not.toBeNull()
  })
  it("copies the code and opens the verification page, preferring the link with the code filled in", async () => {
    const open = vi.fn()
    const writeText = vi.fn(() => Promise.resolve())
    vi.stubGlobal("window", { open, isSecureContext: true })
    vi.stubGlobal("navigator", { clipboard: { writeText } })
    fake.account.current = Result.success(signingInWithCode)
    withCode(Option.some("https://hf.co/oauth/device?user_code=WXYZ-1234"))
    const row = HuggingFaceRow() as ReactElement<RowProps>
    await expect(row.props.onCopyCode()).resolves.toBe(true)
    expect(writeText).toHaveBeenCalledWith("WXYZ-1234")
    row.props.onOpenVerification()
    expect(open).toHaveBeenCalledWith("https://hf.co/oauth/device?user_code=WXYZ-1234", "_blank", "noopener,noreferrer")
    withCode()
    ;(HuggingFaceRow() as ReactElement<RowProps>).props.onOpenVerification()
    expect(open).toHaveBeenLastCalledWith("https://hf.co/oauth/device", "_blank", "noopener,noreferrer")
  })
  it("reports a copy the clipboard refused", async () => {
    vi.stubGlobal("window", { isSecureContext: true })
    vi.stubGlobal("navigator", { clipboard: { writeText: () => Promise.reject(new Error("denied")) } })
    vi.stubGlobal("document", { createElement: () => { throw new Error("no DOM") } })
    fake.account.current = Result.success(signingInWithCode)
    withCode()
    await expect((HuggingFaceRow() as ReactElement<RowProps>).props.onCopyCode()).resolves.toBe(false)
  })
  it("cancels the device sign-in", () => {
    fake.account.current = Result.success(signingInWithCode)
    withCode()
    ;(HuggingFaceRow() as ReactElement<RowProps>).props.onCancel()
    expect(fake.setters.get(fake.atoms.cancel)).toHaveBeenCalledWith({ attemptId: Option.none() })
  })
  it("switches to the signed-in account once the code is entered", () => {
    fake.account.current = Result.success({ _tag: "SignedIn", username: "trg", avatarUrl: Option.none() })
    withCode()
    const html = renderToStaticMarkup(HuggingFaceRow())
    expect(html).toContain("Signed in as trg")
    expect(html).not.toContain("WXYZ-1234")
    expect(button(html, "Sign out")).not.toBeNull()
  })
  it("words a declined or expired device sign-in like the browser one", () => {
    fake.account.current = Result.success({ _tag: "SignedOut", reason: Option.some("Denied") })
    withCode()
    const denied = renderToStaticMarkup(HuggingFaceRow())
    expect(denied).toContain("Sign-in was declined on Hugging Face.")
    expect(denied).not.toContain("WXYZ-1234")
    fake.account.current = Result.success({ _tag: "SignedOut", reason: Option.some("Expired") })
    expect(renderToStaticMarkup(HuggingFaceRow())).toContain("The sign-in wasn’t finished in time. Try again.")
  })
  it("shows no code for another client's device sign-in or an expired code", () => {
    fake.account.current = Result.success(signingInWithCode)
    expect(renderToStaticMarkup(HuggingFaceRow())).not.toContain("Enter this code on Hugging Face")
    withCode(Option.none(), Date.now() - 301_000)
    expect(renderToStaticMarkup(HuggingFaceRow())).not.toContain("WXYZ-1234")
  })
  it("keeps the browser flow and shows no code in the desktop app", () => {
    desktopWindow()
    fake.account.current = Result.success(signingInWithCode)
    withCode()
    expect(view(Result.success({ _tag: "SigningIn", method: "Browser" }))).not.toContain("Enter this code")
    fake.setters.set(fake.atoms.start, vi.fn(() => Promise.resolve(Exit.fail("failed"))))
    fake.account.current = Result.success(signedOut)
    ;(HuggingFaceRow() as ReactElement<RowProps>).props.onSignIn()
    expect(fake.setters.get(fake.atoms.start)).toHaveBeenCalledWith({})
    expect(fake.setters.get(fake.atoms.startDevice)).not.toHaveBeenCalled()
  })
})

describe("gated Hugging Face repositories", () => {
  const signIn = new HuggingFaceAccessRequired({ repository: "google/gemma-4-GGUF", access: "SignIn", message: "PRIVATE" })
  const terms = new HuggingFaceAccessRequired({ repository: "google/gemma-4-GGUF", access: "AcceptTerms", message: "PRIVATE" })
  const actions = (error: HuggingFaceAccessRequired, account: Option.Option<HuggingFaceAccount>) => renderToStaticMarkup(
    <ErrorNotice {...candidateErrorNotice(error)} actions={<HuggingFaceAccessActions error={error} account={account} starting={false} onSignIn={() => {}} onRetry={() => {}} />} />)
  it("asks a signed-out user to sign in, with Try again", () => {
    expect(candidateErrorNotice(signIn)).toMatchObject({ title: "Sign in to Hugging Face to use this model", retryable: true, access: signIn })
    const html = actions(signIn, Option.some(signedOut))
    expect(html).toContain("Sign in to Hugging Face to use this model")
    expect(button(html, "Sign in")).not.toBeNull()
    expect(button(html, "Try again")).not.toBeNull()
    expect(html).not.toContain("PRIVATE")
    expect(html).not.toContain("HF_TOKEN")
  })
  it("waits for the browser while signing in, and leaves only Try again once signed in", () => {
    const waiting = actions(signIn, Option.some({ _tag: "SigningIn", method: "Browser" }))
    expect(waiting).toMatch(/<button[^>]*disabled=""[^>]*>Waiting for the browser…<\/button>/)
    const signedIn = actions(signIn, Option.some({ _tag: "SignedIn", username: "trg", avatarUrl: Option.none() }))
    expect(button(signedIn, "Sign in")).toBeNull()
    expect(button(signedIn, "Try again")).not.toBeNull()
  })
  it("sends a user who must accept terms to the repository on Hugging Face, with Try again", () => {
    const notice = candidateErrorNotice(terms)
    expect(notice).toMatchObject({ title: "Accept this model’s terms on Hugging Face", retryable: true })
    expect(notice.description).toContain("author’s approval")
    const html = actions(terms, Option.some({ _tag: "SignedIn", username: "trg", avatarUrl: Option.none() }))
    expect(html).toMatch(/<a href="https:\/\/huggingface.co\/google\/gemma-4-GGUF" target="_blank"[^>]*>Open on Hugging Face/)
    expect(button(html, "Sign in")).toBeNull()
    expect(button(html, "Try again")).not.toBeNull()
  })
  it("words a download refused for access", () => {
    expect(downloadNotice({ _tag: "HuggingFaceAccessRequired", access: "SignIn" }).title).toBe("Sign in to Hugging Face to use this model")
    expect(downloadNotice({ _tag: "HuggingFaceAccessRequired", access: "AcceptTerms" }).title).toBe("Accept this model’s terms on Hugging Face")
  })
})
