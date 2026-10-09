/**
 * The browser app's files embedded in the release service executable, as `{ path, file }` pairs:
 * the URL path and the embedded file. The release build rewrites this module before compiling; a
 * service run from source serves the `web` build directory instead.
 */
export const embeddedWebApp: ReadonlyArray<{ readonly path: string; readonly file: string }> = []
