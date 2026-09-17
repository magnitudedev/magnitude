# Native window acceptance

Run `acceptance.mjs` with Node and the environment variables:

- `PLAYWRIGHT_MODULE`: absolute path to Playwright's `index.mjs`.
- `MAGNITUDE_TEST_APP`: packaged Electron executable. On Linux use the installed launcher, or a test-copy launcher that retains its installation lease and points at the test executable.
- `CHROME_EVIDENCE`: directory for screenshots and JSON results.
- Windows: `NATIVE_CHROME_SCRIPT` points to `windows-hit-test.ps1` to verify native caption and maximize hit targets at the current display scale.
- Optional `CHROME_MANUAL_CHECK=1`: retain the test window for a minute after automated checks.

The runner uses a separate temporary Magnitude profile on port 11279. It verifies navigation, native control geometry, appearance, minimize/restore, maximize/restore, fullscreen, minimum-size layout, and close-to-hide/reopen behavior. Linux requires an X11 session and `xprop`; the distribution checks may run in Xvfb with a real window manager. It does not validate model inference or claim Wayland coverage. Rendered page captures omit native controls; `*-native.png` captures include Windows controls, but Linux window captures may omit the surrounding window-manager frame (checked separately through `_NET_FRAME_EXTENTS`).

Use the normal Electron build from the desktop package. When constructing a bundle manually, retain the package's ES-module main entry and `.mjs` preload format. Keep platform-specific native binaries and service resources from that platform; do not copy macOS binaries into a Windows or Linux test package.
