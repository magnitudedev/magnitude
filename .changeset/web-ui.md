---
"@magnitudedev/cli": patch
---

commit: c57fcead
author: @thrgreenwald

- Use Magnitude from a web browser. The Magnitude service now serves the full app at `http://<computer>:10100`, the same interface as the desktop app: Discover, Catalog, My Models, Connections, Usage, Status and Settings.
- Open the app from another device on your network. Turn on Network access in Settings, then sign in from the other device's browser with the network access key. You stay signed in until you sign out or the service restarts, and the app warns you before changes that would disconnect your browser (turning off network access, a new key, a different address, or a restart).
- Run Magnitude on a headless machine with `magnitude serve` and manage it entirely from your browser: download and load models, watch usage, connect agents and change settings without the desktop app.
- Settings that belong to the machine running Magnitude now work from every app and browser: agent connections, the model storage folder (with a folder picker inside the app), network access and machine details.
- Updates, launch at login, restart and quit can be controlled from any connected app or browser, not only the desktop window. Controls that a headless server can't offer are hidden there.
- If the browser loses the service, it shows "Can't reach Magnitude" and reconnects on its own when the service comes back.
- The app works in narrow windows and on phones: below tablet width it switches to a top bar and a navigation drawer, and every page reflows to fit.
- Removed the old unused agent web app.
