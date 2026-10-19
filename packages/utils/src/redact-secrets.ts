const HUGGING_FACE_TOKEN = /hf_[A-Za-z0-9_]+/g
const BEARER_CREDENTIAL = /\b(authorization["']?\s*[:=]\s*["']?(?:bearer|basic)\s+)[^\s"',;]+/gi

/** Replaces Hugging Face tokens and `Authorization` credentials in diagnostic text. */
export const redactSecrets = (text: string): string =>
  text.replace(BEARER_CREDENTIAL, "$1***").replace(HUGGING_FACE_TOKEN, "hf_***")
