# Magnitude CLI

Install the Magnitude desktop app from [magnitude.dev](https://magnitude.dev) first.
The app includes the headless CLI and manages its updates.

Open the app once, then open a new terminal:

```sh
magnitude --help
```

The desktop installer provides the command. This launcher package is private and is not published to npm.

This package runs the CLI bundled with your installed desktop app. It does not
install a separate engine, download another CLI, or manage app updates. If the
app is missing, the command tells you where to download it.

The launcher finds the standard macOS, Windows, and Linux desktop installation.
For a custom location, set `MAGNITUDE_DESKTOP_PATH` to the `.app` bundle on macOS
or the desktop executable on Windows and Linux.
