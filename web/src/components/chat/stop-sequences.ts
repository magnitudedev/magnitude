export const formatStopSequences = (sequences: readonly string[]): string =>
  sequences
    .map((sequence) =>
      sequence
        .replace(/\\/g, "\\\\")
        .replace(/\n/g, "\\n")
        .replace(/\r/g, "\\r")
        .replace(/\t/g, "\\t")
    )
    .join("\n")
export const parseStopSequences = (text: string): readonly string[] =>
  text
    .split(/\r?\n/)
    .filter(Boolean)
    .map((sequence) =>
      sequence.replace(
        /\\([nrt\\])/g,
        (_, key: string) => ({ n: "\n", r: "\r", t: "\t", "\\": "\\" }[key]!)
      )
    )
