# Remote access

Magnitude serves its inference API on port 10100 of the computer running it. By default only that computer can reach it.

## Base URLs

- OpenAI-compatible: `http://ADDRESS:10100/inference/v1`
- Anthropic-compatible: `http://ADDRESS:10100/inference/anthropic`

On the same computer, `ADDRESS` is `127.0.0.1` and no API key is needed. Send any value if the client requires one.

## From another device

Enable Network access in Magnitude Settings, or set `network.enabled`, `network.bind`, and `network.apiKey` in the server's configuration file, `~/.magnitude/config.json` (`%USERPROFILE%\.magnitude\config.json` on Windows):

```json
{
  "network": {
    "enabled": true,
    "bind": "100.64.1.5",
    "apiKey": "A_LONG_RANDOM_KEY"
  }
}
```

For `bind`, use an IP address assigned to the server, or omit it to serve every interface. Use a long, randomly generated key. Restart Magnitude after changes. The address can be on a local network or a private network such as Tailscale. Send the key as `Authorization: Bearer KEY` or `x-api-key: KEY`. A person can also open `http://ADDRESS:10100` in a browser on another device and sign in with the same key to use the Magnitude app there; that browser session never authorizes inference requests, and `/rpc` refuses other devices that have not signed in.

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

On the remote computer, install the CLI only (see `magnitude docs headless`), turn on Network access
in its `~/.magnitude/config.json` as above, and run `magnitude serve`. To keep it serving after logout
and across restarts, set it up as a background service that starts at boot, as described in
`magnitude docs headless`; ask your user before doing that. `magnitude status` on the remote computer
shows the addresses other devices can use, never the key.

Without a background service, leave the terminal running `magnitude serve` open. Use another SSH
session for commands such as `magnitude models list` and `magnitude models load <model-id>`. Ctrl+C
stops it. While the desktop app is open on that computer, it serves instead, and `serve` waits until
the app quits.

For direct access, enable Network access and use the server address and API key in your client. To connect through SSH without enabling Network access, open a tunnel on your own computer:

```sh
ssh -N -L 10101:127.0.0.1:10100 user@server
```

Keep the tunnel running and use `http://127.0.0.1:10101/inference/v1` locally. No Magnitude API key
is needed through this loopback tunnel. Model management commands still run on the remote computer.

## Linux servers set up by older versions

A Linux server set up as a system service by an earlier version of Magnitude keeps working after it
updates: it still runs as `magnitude.service` under the `magnitude` account, with the same address, port and key.
Its data and configuration are in `/var/lib/magnitude/.magnitude/`, so its configuration file is
`/var/lib/magnitude/.magnitude/config.json`. Logs: `journalctl -u magnitude`. Other accounts can't use or
control that service with the `magnitude` command: `magnitude status` reports it as another account's.

## Updates on a server

A CLI-only install updates itself while `serve` runs, between requests, and clients reconnect when it
is back. See `magnitude docs headless`.

## WSL

Inside WSL on Windows, `127.0.0.1` is the Linux distribution, not Windows. With WSL mirrored networking, `http://127.0.0.1:10100` works. Otherwise use the Windows host address from `ip route show default | awk '{print $3}'` with Network access on and the API key.

## Docker

The `magnitudedev/magnitude` image runs `magnitude serve` and always accepts connections on every
interface with an API key. `MAGNITUDE_NETWORK_KEY` sets the key, and `MAGNITUDE_ALLOWED_HOSTS` adds host
names, comma-separated. Use the port the container publishes on its host, such as
`http://HOST:10100`, not the container's own `127.0.0.1`. Run commands inside it with
`docker exec <container> magnitude ...`.
