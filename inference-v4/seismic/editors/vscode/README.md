# Seismic for VS Code

Local syntax highlighting, comment toggling, bracket matching, indentation, and
folding for `.seismic` files. No compiler or language server required.

From this directory, package and install:

```sh
npx --yes @vscode/vsce package --allow-missing-repository --skip-license
code --install-extension seismic-language-0.0.3.vsix
```

Alternatively, use **Extensions: Install from VSIX...** in VS Code's command
palette and select the generated file. Nothing is published.

To try the source without packaging, run from this directory:

```sh
code --new-window --extensionDevelopmentPath="$PWD"
```

Open a Seismic source file in that window. Colors follow your current theme.
