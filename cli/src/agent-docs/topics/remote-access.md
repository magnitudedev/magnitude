# Remote access

Magnitude serves its inference API on port 10100 of the computer running it. By default only that computer can reach it.

## Base URLs

- OpenAI-compatible: `http://ADDRESS:10100/inference/v1`
- Anthropic-compatible: `http://ADDRESS:10100/inference/anthropic`

On the same computer, `ADDRESS` is `127.0.0.1` and no API key is needed. Send any value if the client requires one.

## From another device

The user turns on Network access in Magnitude Settings. That setting shows the address to use and an API key. Send the key as `Authorization: Bearer KEY` or `x-api-key: KEY`. Only inference is available from other devices; model management and `/rpc` are not.

Check the connection:

```sh
curl -H "Authorization: Bearer KEY" http://ADDRESS:10100/inference/v1/models
```

| Response | Meaning |
| --- | --- |
| Connection refused | Network access is off, or the chosen address does not include this network. |
| 401 | Missing or changed API key. |
| 421 | Hostname not accepted. Use an IP address, or the user adds the name to `network.allowedHosts` in `~/.magnitude/config.json`. |

## Headless server over SSH

On the remote computer, run `magnitude serve` and leave that terminal running. Use another SSH
session for commands such as `magnitude models status` and `magnitude models load <model-id>`.

To reach inference from your own computer without enabling Network access, open a tunnel:

```sh
ssh -N -L 10101:127.0.0.1:10100 user@server
```

Keep the tunnel running and use `http://127.0.0.1:10101/inference/v1` locally. No Magnitude API key
is needed through this loopback tunnel. Model management commands still run on the remote computer.
Ctrl+C stops the foreground server. Opening Magnitude Desktop on that computer transfers ownership
to Desktop and ends the foreground command.

An available update never interrupts the server. When it reports a prepared update, stop the server
and run `magnitude serve` again to install it before serving. A failed installation requires an
explicit retry with `magnitude update install` while the server is stopped.

## WSL

Inside WSL on Windows, `127.0.0.1` is the Linux distribution, not Windows. With WSL mirrored networking, `http://127.0.0.1:10100` works. Otherwise use the Windows host address from `ip route show default | awk '{print $3}'` with Network access on and the API key.
