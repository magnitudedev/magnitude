/**
 * Lets a reader that closes this process's stdout early (`| head`) end the output without failing the command.
 * Bun reports the closed pipe as an `error` event on the stream, which unhandled prints a stack trace and exits 1.
 */
export const ignoreClosedOutput = (stream: NodeJS.WritableStream): void => {
  stream.on("error", (error: NodeJS.ErrnoException) => {
    if (error.code !== "EPIPE") throw error
  })
}
