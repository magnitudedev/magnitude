# Remote access

Magnitude serves its inference API on port 10100 of the computer running it. By default only that computer can reach it.

## Base URLs

- OpenAI-compatible: `http://ADDRESS:10100/inference/v1`
- Anthropic-compatible: `http://ADDRESS:10100/inference/anthropic`

On the same computer, `ADDRESS` is `127.0.0.1` and no API key is needed. Send any value if the client requires one.

## From another device

Enable Network access in Magnitude Settings, or set `network.enabled`, `network.bind`, and `network.apiKey` in the server's configuration file: `~/.magnitude/config.json`, or `/var/lib/magnitude/config.json` for a Linux server set up with `magnitude server setup` (which turns Network access on itself). Use an IP address assigned to the server and a long, randomly generated key. Restart Magnitude after changes. The address can be on a local network or a private network such as Tailscale. Send the key as `Authorization: Bearer KEY` or `x-api-key: KEY`. A person can also open `http://ADDRESS:10100` in a browser on another device and sign in with the same key to use the Magnitude app there; that browser session never authorizes inference requests, and `/rpc` refuses other devices that have not signed in.

Check the connection:

```sh
curl -H "Authorization: Bearer KEY" http://ADDRESS:10100/inference/v1/models
```

| Response | Meaning |
| --- | --- |
| Connection refused | Network access is off, or the chosen address does not include this network. |
| 401 | Missing or changed API key. |
| 421 | Hostname not accepted. Use an IP address, or the user adds the name to `network.allowedHosts` in that configuration file. |

## Remote inference server

On Linux and macOS, `magnitude server setup` on the remote computer runs Magnitude as a service: it
starts at boot, keeps running after logout, restarts if it crashes, turns on Network access, creates
an API key, and prints the addresses to use. Run it as the person, not with `sudo`; it asks for a
password once. `magnitude status` shows the addresses later, never the key. `magnitude server remove`
stops and removes the service and keeps its data.

- Linux: the service is `magnitude.service`, running as the `magnitude` system account. Its models,
  settings and logs are in `/var/lib/magnitude`, and its configuration is
  `/var/lib/magnitude/config.json`, not `~/.magnitude/config.json`. Logs: `journalctl -u magnitude`.
- macOS: the service is the `dev.magnitude.server` LaunchDaemon, running as the person who set it up
  and using their `~/.magnitude/config.json`. Logs: `~/.magnitude/logs/server.log`.
- Windows has no server mode. Wrap `magnitude serve` in a service wrapper such as NSSM, running as the
  account that installed Magnitude, and start the Magnitude command on PATH
  (`%LOCALAPPDATA%\Programs\Magnitude CLI\magnitude.exe serve`), not the application's own
  `resources\magnitude.exe`.

Without the service, run `magnitude serve` on the remote computer and leave that terminal running.
Use another SSH session for commands such as `magnitude models status` and
`magnitude models load <model-id>`. Ctrl+C stops it. Opening Magnitude Desktop on that computer
takes over from a hand-run `serve` and ends it; while a server set up with `magnitude server setup`
runs, Desktop instead offers to keep the server or stop it.

For direct access, enable Network access and use the server address and API key in your client. To connect through SSH without enabling Network access, open a tunnel on your own computer:

```sh
ssh -N -L 10101:127.0.0.1:10100 user@server
```

Keep the tunnel running and use `http://127.0.0.1:10101/inference/v1` locally. No Magnitude API key
is needed through this loopback tunnel. Model management commands still run on the remote computer.

## Updates on a server

On Linux and macOS, the service and a hand-run `serve` install downloaded updates by themselves
between requests, when they can do so without a password: on Linux when `sudo` asks for none, on
macOS when the person can write to the Applications folder. Clients reconnect when the server is
back. Otherwise `serve` prints that the update is downloaded: stop it, run `magnitude update install`,
and start it again. On Windows, stop the wrapped service, run `magnitude update install`, then start
the service again.

## WSL

Inside WSL on Windows, `127.0.0.1` is the Linux distribution, not Windows. With WSL mirrored networking, `http://127.0.0.1:10100` works. Otherwise use the Windows host address from `ip route show default | awk '{print $3}'` with Network access on and the API key.
