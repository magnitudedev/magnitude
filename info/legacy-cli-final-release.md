# Final legacy CLI release

The final CLI release displays a desktop migration notice before setup, pauses for three seconds
in interactive terminals, and then continues normally. The npm launcher still installs and works.

After the final release is explicitly approved and published, mark the legacy npm package deprecated:

```sh
npm deprecate '@magnitudedev/cli@*' '
========================================================
MAGNITUDE HAS MOVED TO A FREE, OPEN SOURCE DESKTOP APP.

THIS LEGACY CLI WILL NO LONGER RECEIVE UPDATES.

DOWNLOAD THE DESKTOP APP:
https://magnitude.dev
========================================================
'
```

This is a separate registry mutation, not a release script. Do not execute it before launch approval.
It covers existing legacy versions as well as the final release without removing any packages.
Npm controls the warning's layout and placement during installation; it is not a custom postinstall
banner. Preserve the literal newlines in the command above: npm displays the separators and blank
lines, but prefixes each line with `npm warn deprecated`. Verified with a real npm installation
against a local registry. Successful lifecycle script output is normally hidden by npm, so a postinstall script would
not reliably display the message. Users suppressing npm warnings will also suppress this notice.

References: [npm deprecation](https://docs.npmjs.com/cli/v11/commands/npm-deprecate/)
and [foreground scripts](https://docs.npmjs.com/cli/v11/commands/npm-install/#foreground-scripts).
