# Headless use

Run Magnitude without the desktop app: on a server, over SSH, or to serve other apps on this computer.

## Installs

Magnitude installs in one of two ways:

- **Desktop app**, for a laptop or desktop where a person uses the app. It includes the CLI. Download it
  from https://magnitude.dev/download, or install it from a terminal:

  ```sh
  curl -fsSL https://magnitude.dev/install.sh | sh -s -- --desktop
  ```

  ```powershell
  & ([scriptblock]::Create((irm https://magnitude.dev/install.ps1))) -Desktop
  ```

- **CLI only**, for a server, or any machine without the desktop app:

  ```sh
  curl -fsSL https://magnitude.dev/install.sh | sh
  ```

  ```powershell
  irm https://magnitude.dev/install.ps1 | iex
  ```

The CLI-only install needs no root or administrator rights. On macOS and Linux it installs into
`~/.magnitude/cli/<version>/` and puts the `magnitude` command at `~/.magnitude/bin/magnitude`, adding
`~/.magnitude/bin` to PATH in the shell startup files; open a new shell to use it. On Windows it
installs into `%USERPROFILE%\.magnitude\cli\<version>\` and the command is
`%LOCALAPPDATA%\Programs\Magnitude CLI\magnitude.exe`, added to the user PATH.

If the desktop app is already installed, the CLI-only installer says so and changes nothing: the app
already includes the CLI.

## Serving

```sh
magnitude serve
```

`magnitude serve` runs the Magnitude service in the foreground until Ctrl+C. It serves the inference
API and the Magnitude app on `http://127.0.0.1:10100`, so by default only this computer can reach it:

- OpenAI-compatible: `http://127.0.0.1:10100/inference/v1`
- Anthropic-compatible: `http://127.0.0.1:10100/inference/anthropic`

While it serves, `magnitude app open` opens the Magnitude app in this computer's browser at
`http://127.0.0.1:10100`, or prints that address where there is no browser. It is the same app as the
desktop app, without one-click Connect in Connections.

A loaded model is unloaded after an hour without requests, or when the system runs short of memory;
the next request, or `magnitude models load`, loads it again.

Run other commands, such as `magnitude status`, `magnitude models recommend` and
`magnitude models pull <model-id>`, in another shell while it serves. They use the running service
and never start one.

To reach it from other devices, set `network.enabled`, `network.bind` and `network.apiKey` in
`~/.magnitude/config.json` (`%USERPROFILE%\.magnitude\config.json` on Windows) and restart `serve`.
See `magnitude docs remote-access` for the settings, an SSH tunnel alternative, and troubleshooting.

### The desktop app takes over

Only one Magnitude serves on a computer, and the desktop app always wins. While the desktop app is
open, it serves, and any `magnitude serve`, in a terminal or a background service and in any
account, prints that another Magnitude app is serving and waits. When the app quits, `serve` takes
over again by itself.

## Running it in the background

Ask your user before setting this up. A background service keeps the inference API running, and
using memory and the GPU while a model is loaded, until it is removed.

Each definition below starts `serve` at boot, with nobody logged in, and runs it as the user with
their `~/.magnitude`. Every one calls the `magnitude` command (`~/.magnitude/bin/magnitude`, or
`%LOCALAPPDATA%\Programs\Magnitude CLI\magnitude.exe` on Windows), never a path inside
`~/.magnitude/cli/<version>/`. That keeps it working across updates, and on macOS and Windows a
later desktop install runs the app's CLI through the same definition (on Linux, see Switching to the
desktop app). Nothing needs to change when the desktop app is open: the service waits while the app
serves and resumes when it quits.

Set up only one of these per computer. Check that it serves with `magnitude status`.

### macOS

A LaunchDaemon runs `serve` as the user through `UserName`. launchd does not expand `~` or `$HOME`
inside a plist, so the heredoc below writes the absolute home path. It needs `sudo`; run the
commands as the user, not in a root shell, so `$HOME` and `id -un` name them:

```sh
mkdir -p ~/.magnitude/logs
sudo tee /Library/LaunchDaemons/dev.magnitude.serve.plist > /dev/null <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>dev.magnitude.serve</string>
  <key>ProgramArguments</key>
  <array><string>$HOME/.magnitude/bin/magnitude</string><string>serve</string></array>
  <key>UserName</key><string>$(id -un)</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key><string>$HOME</string>
    <key>PATH</key><string>/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>WorkingDirectory</key><string>$HOME</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>$HOME/.magnitude/logs/serve.log</string>
  <key>StandardErrorPath</key><string>$HOME/.magnitude/logs/serve.log</string>
</dict>
</plist>
EOF
sudo chown root:wheel /Library/LaunchDaemons/dev.magnitude.serve.plist
sudo chmod 644 /Library/LaunchDaemons/dev.magnitude.serve.plist
sudo launchctl bootstrap system /Library/LaunchDaemons/dev.magnitude.serve.plist
```

With FileVault on, nothing starts after a restart until someone unlocks the disk. After editing the
plist, run the `bootout` command below and then `bootstrap` again.

- Restart: `sudo launchctl kickstart -k system/dev.magnitude.serve`
- Logs: `~/.magnitude/logs/serve.log`, and the service's own log at `~/.magnitude/logs/service.log`
- Stop and remove:

  ```sh
  sudo launchctl bootout system/dev.magnitude.serve
  sudo rm /Library/LaunchDaemons/dev.magnitude.serve.plist
  ```

A Mac set up as a server by an earlier version of Magnitude has a `dev.magnitude.server`
LaunchDaemon. It keeps working. To remove it:
`sudo launchctl bootout system/dev.magnitude.server && sudo rm /Library/LaunchDaemons/dev.magnitude.server.plist`.

### Linux

A systemd user unit runs `serve` as the user, and lingering starts it at boot and keeps it running
without anyone logged in:

```sh
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/magnitude.service <<'EOF'
[Unit]
Description=Magnitude

[Service]
ExecStart=%h/.magnitude/bin/magnitude serve
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
EOF
sudo loginctl enable-linger $USER
systemctl --user daemon-reload && systemctl --user enable --now magnitude
```

- Status: `systemctl --user status magnitude`
- Logs: `journalctl --user -u magnitude`
- Restart: `systemctl --user restart magnitude`
- Stop and remove:

  ```sh
  systemctl --user disable --now magnitude
  rm ~/.config/systemd/user/magnitude.service
  systemctl --user daemon-reload
  ```

  If linger was turned on only for this, `sudo loginctl disable-linger $USER`.

Over SSH, `systemctl --user` needs the user's session bus. If it reports that it can't connect, log in
over SSH directly as the user (not through `su` or `sudo -u`).

### Windows

A scheduled task runs `serve` as the account that installed Magnitude. From an administrator
PowerShell, set `$account` to that account (`COMPUTERNAME\user`, or `DOMAIN\user`) and `$userProfile`
to its profile folder:

```powershell
$account = "$env:COMPUTERNAME\alice"
$userProfile = "C:\Users\alice"
# A task that runs before sign-in logs on as a batch job; Performance Log Users holds that right by default.
Add-LocalGroupMember -Group "Performance Log Users" -Member $account -ErrorAction SilentlyContinue
$action = New-ScheduledTaskAction -Execute "$userProfile\AppData\Local\Programs\Magnitude CLI\magnitude.exe" -Argument "serve"
$trigger = New-ScheduledTaskTrigger -AtStartup
# Checks every minute and starts serve again if it has exited; a running serve is left alone.
$restart = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 1)
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
$password = (Get-Credential $account).GetNetworkCredential().Password
Register-ScheduledTask -TaskName "Magnitude" -Action $action -Trigger $trigger, $restart -Settings $settings -User $account -Password $password -RunLevel Limited
Start-ScheduledTask -TaskName "Magnitude"
```

The task keeps the account's password, entered in a prompt rather than typed on the command line,
because Windows runs a task with nobody signed in only with its password. A standard account also
needs the right to log on as a batch job; the group above grants it (or grant "Log on as a batch job"
in Local Security Policy). After the account's password changes, register the task again.

If other computers need to reach it, allow inbound TCP port 10100 in Windows Defender Firewall from an
administrator PowerShell. Cloud servers usually have a Public network profile, so the rule covers every
profile; set `-RemoteAddress` to the addresses of your clients to admit only them:

```powershell
New-NetFirewallRule -DisplayName "Magnitude" -Direction Inbound -Protocol TCP -LocalPort 10100 -Action Allow -Profile Any -RemoteAddress Any
```

- Status: `Get-ScheduledTask -TaskName "Magnitude" | Get-ScheduledTaskInfo`, and `magnitude status`
- Logs: `%USERPROFILE%\.magnitude\logs\service.log`
- Stop for a while: `Disable-ScheduledTask -TaskName "Magnitude"`, then `Stop-ScheduledTask -TaskName "Magnitude"`;
  `Enable-ScheduledTask -TaskName "Magnitude"` turns it back on.
- Stop and remove, from an administrator PowerShell:

  ```powershell
  Stop-ScheduledTask -TaskName "Magnitude"
  Unregister-ScheduledTask -TaskName "Magnitude" -Confirm:$false
  ```

## Updates

A CLI-only install updates itself while `magnitude serve` runs. It downloads a new release in the
background, waits until no inference request is running, installs it beside the current version, and
restarts in place. Clients reconnect when it is back, and a background service keeps running through
the update. If a new version fails, it is held back and the current one keeps serving.

To stop the automatic downloads, set `"autoDownloadUpdates": false` in `~/.magnitude/config.json`.
`serve` then only prints that an update is available. Install one on demand with:

```sh
magnitude update install
```

Without a running `serve`, nothing updates by itself; `magnitude update install` still works.

## Switching to the desktop app

Install the desktop app from https://magnitude.dev/download, or rerun the installer with `--desktop`
(`-Desktop` on Windows), as shown in Installs above. The app replaces the CLI-only install: it removes
`~/.magnitude/cli` and its CLI becomes the `magnitude` command. Models, settings and network access
stay, since both use `~/.magnitude`. A background service set up above keeps working on every OS: it
calls the `magnitude` command, which is now the app's CLI, and it waits while the app is open.

On Linux the desktop package provides `/usr/bin/magnitude`. With `--desktop`, the installer points
`~/.magnitude/bin/magnitude` at it straight away. With a downloaded package, the CLI-only copy keeps
running until the app is first opened; then `~/.magnitude/bin/magnitude` points at `/usr/bin/magnitude`.

## Removing the CLI-only install

1. Stop and remove any background service, using the removal commands for that service above.
2. Delete the install and the command.

   macOS and Linux:

   ```sh
   rm -rf ~/.magnitude/cli ~/.magnitude/bin/magnitude && rmdir ~/.magnitude/bin 2>/dev/null
   for f in "${ZDOTDIR:-$HOME}/.zprofile" "${ZDOTDIR:-$HOME}/.zshrc" ~/.bash_profile ~/.bash_login ~/.profile ~/.bashrc "${XDG_CONFIG_HOME:-$HOME/.config}/fish/config.fish"; do
     [ -f "$f" ] && sed -i.bak '/# >>> Magnitude CLI PATH >>>/,/# <<< Magnitude CLI PATH <<</d' "$f"
   done
   ```

   The loop removes the PATH block the installer added, marked by `# >>> Magnitude CLI PATH >>>`, and
   leaves a `.bak` copy of each file it changed.

   Windows PowerShell:

   ```powershell
   $dir = "$env:LOCALAPPDATA\Programs\Magnitude CLI"
   Remove-Item -Recurse -Force $dir, "$env:USERPROFILE\.magnitude\cli"
   # Edit the registry value directly so it keeps its type and unexpanded entries such as %USERPROFILE%.
   $key = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey("Environment", $true)
   $kind = $key.GetValueKind("Path")
   $path = $key.GetValue("Path", "", "DoNotExpandEnvironmentNames") -split ";" |
     Where-Object { $_ -and [Environment]::ExpandEnvironmentVariables($_).TrimEnd("\") -ne $dir }
   $key.SetValue("Path", ($path -join ";"), $kind)
   $key.Close()
   ```

3. Models, settings, keys and logs stay in `~/.magnitude` (`%USERPROFILE%\.magnitude` on Windows), so
   a later install picks them up. Delete that folder too only if the user wants them gone.

Don't run these steps when the desktop app is installed: its uninstaller removes it.
