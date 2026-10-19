import type { HuggingFaceAccess, HuggingFaceAccount, HuggingFaceSignedOutReason } from "@magnitudedev/sdk"

/** What a gated repository needs, in the words every client shows. */
export const huggingFaceAccessMessage = (access: HuggingFaceAccess): string =>
  access === "SignIn"
    ? "Sign in to Hugging Face to use this model."
    : "Accept this model's terms on Hugging Face, then try again. Some models also need the author's approval."

/** Why the last sign-in didn't finish or the account was signed out. */
export const huggingFaceSignedOutMessage = (reason: HuggingFaceSignedOutReason): string => {
  switch (reason) {
    case "Denied": return "Sign-in was declined on Hugging Face."
    case "Expired": return "The sign-in wasn’t finished in time. Try again."
    case "Failed": return "Magnitude couldn’t sign in to Hugging Face. Try again."
    case "SessionExpired": return "Your Hugging Face sign-in expired. Sign in again."
  }
}

/** One line for the account: who is signed in, a sign-in in progress, or why it is signed out. */
export const huggingFaceAccountSummary = (account: HuggingFaceAccount): string => {
  switch (account._tag) {
    case "SignedIn": return `Signed in as ${account.username}`
    case "SigningIn": return account.method === "Browser" ? "Waiting for the browser…" : "Waiting for the code to be entered…"
    case "SignedOut": return account.reason._tag === "Some"
      ? huggingFaceSignedOutMessage(account.reason.value)
      : "Not signed in. Sign in for gated models, private repositories and higher download limits."
  }
}
