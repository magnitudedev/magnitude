# @magnitudedev/cli

## 0.2.11

### Patch Changes

- [`2e321c6`](https://github.com/magnitudedev/magnitude/commit/2e321c6d221edc0717a751270b480dbe1f08403c) Thanks [@thrgreenwald](https://github.com/thrgreenwald), [@anerli](https://github.com/anerli)! - Run custom models: any GGUF of a model family Magnitude supports, not just catalog models, including other quantizations, finetunes, merges and uncensored versions.

  - Search Hugging Face and add models from the desktop app, the web app or the CLI (`magnitude catalog search`, `inspect` and `pull`). Every version shows its size and whether it fits your computer, can't run (with the reason) or is gated, before you download it.
  - Sign in to Hugging Face once to download gated models; Magnitude says when a model needs sign-in, its terms accepted, or the author's approval.
  - Models already on your computer are found and used in place, without copying: the Hugging Face cache, LM Studio, the llama.cpp cache, Ollama, and folders you add. Found models can be hidden without touching their files, and a catalog model already on your computer is used instead of downloaded again.
  - Each model gets the best drafter known to work with it, when there is one: from its own repository or folder, the matching catalog model, or its built-in MTP heads. You can choose a different drafter or none.
  - Models that can't run are listed with the reason, such as an architecture without kernels yet or an unsupported weight type.

- [`2e321c6`](https://github.com/magnitudedev/magnitude/commit/2e321c6d221edc0717a751270b480dbe1f08403c) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - Discover and the model catalog are now one page.

  - Model cards show one clear status and an estimated download time, and catalog and custom models share the same details view.
  - Loading and optimizing show live progress.
  - Service status and model memory moved into Settings.

- [`2e321c6`](https://github.com/magnitudedev/magnitude/commit/2e321c6d221edc0717a751270b480dbe1f08403c) Thanks [@thrgreenwald](https://github.com/thrgreenwald), [@anerli](https://github.com/anerli)! - Faster generation.

  - On x86 CPUs with AVX2 or VNNI, k-quant, Q8 and 4-bit codebook weights run faster.
  - On NVIDIA GPUs, IQ2_XXS weights read faster, and models that stream a token table decode faster: Nemotron 3.5 Bolt on a GB10 went from 36.0 to 41.9 tok/s with drafting off.
  - Drafting decides from the time recent steps actually took, so one slow step no longer turns it off, and it resumes once it pays off again.
  - AMD GPUs on Linux (Vulkan, RADV) prepare a model's programs in a fraction of the time and memory.
  - Chat templates that scan the whole conversation for every message no longer take seconds to render, and short follow-up turns resume from the cached conversation.
  - A catalog model's one-time optimization waits until the GPU is free, and loading a model you want takes priority over it.

- [`2e321c6`](https://github.com/magnitudedev/magnitude/commit/2e321c6d221edc0717a751270b480dbe1f08403c) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - Fix installation and platform issues.

  - On Windows 10, upgrades no longer fail to replace the installed app.
  - The desktop app no longer takes over a Magnitude server running as a Windows service, and says why.
  - Hermes connections on Windows are written where Hermes reads them.
  - Fix installer readiness checks, macOS onboarding, and verification error messages.
  - CLI commands piped into a reader that stops early end quietly instead of crashing.

- [`2e321c6`](https://github.com/magnitudedev/magnitude/commit/2e321c6d221edc0717a751270b480dbe1f08403c) Thanks [@thrgreenwald](https://github.com/thrgreenwald), [@anerli](https://github.com/anerli)! - More accurate memory planning and more reliable loading.

  - Fit checks count each backend's own memory costs, so a model shown as fitting loads, and a long conversation's cached prompt no longer grows past what was planned.
  - Models too large for a small GPU show as not fitting instead of failing to load.
  - Magnitude serves on the discrete GPU when your computer also has an integrated one, and shows which device it's using.
  - Requested downloads, drafters and projectors included, resume after a restart, and a cancelled download stays cancelled.
  - A large image no longer holds a model running on the CPU for minutes.

- [`2e321c6`](https://github.com/magnitudedev/magnitude/commit/2e321c6d221edc0717a751270b480dbe1f08403c) Thanks [@thrgreenwald](https://github.com/thrgreenwald), [@anerli](https://github.com/anerli)! - Fix incorrect output and load failures for several models.

  - Reasoning works with LFM2.5 Instruct and Thinking models, Gemma 4 thinks unless thinking is turned off, and a response that ends inside its reasoning keeps that reasoning instead of failing.
  - Default sampling is now temperature 1.0, top_p 0.95 and top_k 20 on the OpenAI and Anthropic APIs alike. The OpenAI APIs used 0.8 and 0.95, and the Anthropic API 1.0 and 1.0, with no top_k on either. The Anthropic API now accepts `top_k`.
  - Finetunes run with their base model's vision projector, and Qwen 3.5 models without built-in prediction heads run.
  - Fix Vulkan sampling probabilities, Vulkan weight formats failing to compile, and a Metal weight-packing bug.

- [`ac67c47`](https://github.com/magnitudedev/magnitude/commit/ac67c472d049e6899a0ba9782c39ca17d8ff8671) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - Add a native Chat tab with local model streaming, performance metrics, reasoning controls, generation settings, and SQLite conversation history.

  Render code blocks with language labels, syntax highlighting, and individual copy actions, and improve Markdown headings and nested spacing.

- [`2e321c6`](https://github.com/magnitudedev/magnitude/commit/2e321c6d221edc0717a751270b480dbe1f08403c) Thanks [@anerli](https://github.com/anerli)! - Add Nemotron 3.5 Bolt 74B-A9B, Nemotron 3.5 Super VL 120B-A12B with vision, and Qwen 3.8 Flash Next NVFP4 to the catalog.

  - Nemotron 3.5 models draft with LiLiCorr, a new drafting method ranked above DFlash2. On a GB10, Bolt decodes at 55.7 tok/s against llama.cpp's 51.6.
  - NVFP4 models that carry NVIDIA's activation scales, such as Qwen 3.8 Flash Next NVFP4 and Super VL's NVFP4 variant, run on FP4 tensor cores on GPUs that have them. On a GB10, Super VL NVFP4 prefills at 826 tok/s against llama.cpp's 552.
  - Nemotron 3 Super and Ultra can now run as custom models.

- [`2e321c6`](https://github.com/magnitudedev/magnitude/commit/2e321c6d221edc0717a751270b480dbe1f08403c) Thanks [@anerli](https://github.com/anerli), [@thrgreenwald](https://github.com/thrgreenwald)! - Speed estimates in Discover are now based on measurements of your own hardware instead of a generic memory-bandwidth figure, so they track real performance more closely. The measurement runs once, in the background, and never while a model is in use.

## 0.2.10

### Patch Changes

- [`15ff695`](https://github.com/magnitudedev/magnitude/commit/15ff69513ec4ce5a2420cbdaf5bde9657493f8b4) Thanks [@anerli](https://github.com/anerli)! - Fix memory planning on unified-memory and integrated GPUs.

  - On NVIDIA GB10 and Apple Silicon, the one-time optimization counts the memory the loaded model already uses instead of overcommitting it ([#199](https://github.com/magnitudedev/magnitude/issues/199), [#206](https://github.com/magnitudedev/magnitude/issues/206)).
  - Integrated GPUs such as AMD Strix Halo use their shared memory as well as the dedicated carve-out, instead of stopping when the carve-out is full ([#207](https://github.com/magnitudedev/magnitude/issues/207)).
  - The optimization retries an allocation the GPU refuses instead of failing.
  - The app lists each GPU once and shows the backend Magnitude uses for it.

- [`7f02906`](https://github.com/magnitudedev/magnitude/commit/7f029060a37ed1f7884db038222dc8f36430a08e) Thanks [@anerli](https://github.com/anerli)! - Support models with IQ2_XXS weights on every backend, and fix models with IQ2_XS weights failing to load on a Mac ([#195](https://github.com/magnitudedev/magnitude/issues/195)).

- [`7ec9327`](https://github.com/magnitudedev/magnitude/commit/7ec93279cbd5aa32ae9aa7c6078a1248aec181d9) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - Fix model loads failing on Linux machines and virtual machines ([#196](https://github.com/magnitudedev/magnitude/issues/196)).

  - Memory held by the file cache counts as available, so loads no longer fail for lack of memory that the system could free.
  - A real memory shortage at startup is reported as one that can be retried, instead of failing the load partway through.
  - A CPU without the instructions the engine needs is reported plainly before loading starts.

- [`151bf9a`](https://github.com/magnitudedev/magnitude/commit/151bf9aefcbda14ffa4b420b40a25830fbe7af7b) Thanks [@anerli](https://github.com/anerli)! - Fix models with a Q8_0 output head failing to load, including MiniCPM5 1B Q8 and, with drafting off, Qwen3.6 35B-A3B and Qwen3.5 9B Q6. Faster output for large vocabularies can be selected again during the one-time optimization.

- [`1f54d5b`](https://github.com/magnitudedev/magnitude/commit/1f54d5b7e871aff1d1ae276d6a7cc3a3553baabf) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - The web app's model status explains that a model loads automatically on the first request, and the Update button explains what an update brings.

- [`fc5e2e8`](https://github.com/magnitudedev/magnitude/commit/fc5e2e8adebfe9ab84ccbc95cdf5886c282ace40) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - Fix the RPM package conflicting with other installed applications over shared build IDs ([#200](https://github.com/magnitudedev/magnitude/issues/200)).

## 0.2.9

### Patch Changes

- [`5bbdd73`](https://github.com/magnitudedev/magnitude/commit/5bbdd73809a17be7f4dae2e0aed25eac800eb158) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - Connect Claude Code without a Claude login. Requests, including those from its built-in subagents, use the loaded local model. If you sign in later, reconnect to use your own login for Claude models.

- [`195481a`](https://github.com/magnitudedev/magnitude/commit/195481a154c389cfd4b1c9c00d7c08baf3209408) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - Clarify Discover's speed-versus-quality preference under "Recommended for your machine", and update the README and documentation to explain hardware-specific kernel tuning and the performance comparison with Ollama.

- [`ff42cdb`](https://github.com/magnitudedev/magnitude/commit/ff42cdb2a12d1c33a61bbaff1a97525f81829475) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - Fix installation, update and reliability problems found in release testing.

  - Linux packages install on RHEL 9 and other systems with glibc 2.34, and the Ubuntu and Debian package no longer pulls a KDE desktop onto headless machines.
  - The Linux desktop app no longer crashes when it restarts to apply a setting.
  - On Windows, the app window comes back after Restart to update, and the Magnitude app and service show Magnitude's name, publisher and version in firewall prompts, Task Manager and file properties.
  - On a Mac, another account's leftover installation lock no longer stops the app from opening, and installing says plainly when another account owns the app or left an update behind.
  - An update that installs but never starts is reported as failed by the desktop app too, and isn't installed again automatically. `magnitude update status` and Settings show a failed update as failed, update errors are readable sentences, and `magnitude update check` prints its result. Linux updates wait for another package manager to finish instead of failing.
  - A failed inference engine download is reported in one plain sentence instead of an internal error, and when port 10100 is taken, Magnitude says whether another account's Magnitude or another program holds it.
  - Cancelling a model download removes its partial files, and the free-space check counts a model and its separate draft together. A model folder Magnitude can't write to is refused, and an unusable saved folder no longer stops Magnitude from starting.
  - `magnitude status` shows the address the service actually listens on, and `magnitude connections list` shows agents you never connected as not connected.

- [`00f2045`](https://github.com/magnitudedev/magnitude/commit/00f20455cb9dc643da59edbc2115c520984300b9) Thanks [@anerli](https://github.com/anerli)! - Improve inference performance through better kernel tuning and token packing.

  - The one-time optimization spends its time on the kernels that account for the most measured work, within the existing time budget. This improves generation speed, including Gemma 4 on Apple Silicon.
  - Token packing on Apple Silicon now supports 8-bit weight formats, including DFlash draft models, and improves packing accuracy for existing formats. This also speeds up prompt processing.

- [`ce324d2`](https://github.com/magnitudedev/magnitude/commit/ce324d23b7f85f3f1de806900d97b5f698e7e4bf) Thanks [@anerli](https://github.com/anerli)! - Add Liquid LFM2.5 24B-A2B to the catalog in Q4, Q5, Q6 and Q8. This reasoning model supports local agents, coding and tool use.

- [`707b012`](https://github.com/magnitudedev/magnitude/commit/707b012cba52a043a1eae8a557893572c6353df8) Thanks [@anerli](https://github.com/anerli)! - Add MiniCPM-V 4.7 1B and 35B-A3B in Q4, Q5, Q6 and Q8, with image understanding and DSpark speculative decoding. Both models include their 4x vision projectors and Q8 drafters. Image processing reuses identical crops and checks memory capacity before allocating resized images; 16x visual compression is not supported.

- [`f462efd`](https://github.com/magnitudedev/magnitude/commit/f462efd6b822042ea153efe06306050f264931a1) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - Improve model memory management and loading reliability.

  - Keep loaded models' GPU memory resident through idle gaps on Macs, avoiding the delay of wiring it back into memory when the next request arrives.
  - Fix large model loads being rejected because of brief memory pressure caused by the load itself. Memory protection now responds to sustained thrashing or swapping.
  - Refuse to load a model assessed as not fitting before unloading the current model, and disable its Load action in the desktop.
  - Fix NVIDIA GPU capacity being underestimated on Linux when other programs hold GPU memory at startup, so model recommendations and fit assessments use the device's usable capacity.

- [`539c333`](https://github.com/magnitudedev/magnitude/commit/539c3333864bbd19e1db9030821802eece049e46) Thanks [@anerli](https://github.com/anerli)! - Add Qwen3.8 Flash Next in Q2, Q4, Q6 and Q8, with its shared MTP drafting head for faster generation. Its large n-gram tables stream from storage instead of being held whole in memory. Also add a smaller Q2 variant of Qwen3.6 35B A3B, with DFlash speculative decoding.

- [`33907ee`](https://github.com/magnitudedev/magnitude/commit/33907ee513b7291c1227ebdfdbefbd80b187e353) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - - Run Magnitude as a server. `magnitude server setup` makes it a background service that starts at boot, keeps running after you log out, restarts if it crashes, updates itself, and serves browsers and agents on other computers. `magnitude server remove` stops it and keeps your models and settings.

  - Linux: a systemd service running as a dedicated `magnitude` system account, with its data in `/var/lib/magnitude`. On Fedora and other firewalld systems, setup opens port 10100 for you.
  - macOS: a background service that runs as you, starts before anyone logs in, and keeps using your models and settings in `~/.magnitude`.
  - Windows: the remote server guide shows how to run `magnitude serve` as a Windows service with NSSM.
  - Install from a terminal: `curl -fsSL https://magnitude.dev/install.sh | sh` on macOS and Linux, or `irm https://magnitude.dev/install.ps1 | iex` in Windows PowerShell. On Linux the installer asks once, up front, whether to run Magnitude as a server (default no, skipped automatically after 60 seconds or when there's no terminal), then installs without further questions. It doesn't ask for a password when your account already has password-free sudo, refreshes the package lists so it works on fresh Ubuntu and Debian cloud images, waits for another package manager to finish instead of failing, refuses unsupported systems before downloading anything, and says which version it installed or upgraded from.
  - Servers update themselves, including on Ubuntu 25.10 and later, where sudo is sudo-rs: a downloaded update installs when Magnitude is idle, never in the middle of a request, and the server comes back on the new version on its own. If an update fails to install, the server keeps running the current version and won't keep retrying it; a newer release installs normally, and `magnitude update check` retries on demand. If a new version installs but fails to start, the server reports it, won't install that version again on its own, and keeps checking so a fixed release reaches it.
  - Updates never install at startup any more on any platform: Magnitude always starts right away, and tells you when an update is ready (Restart in the desktop app, or `magnitude update install`).
  - On Linux desktops, installing an update with Restart no longer asks for your password.
  - When Magnitude runs as a server, `magnitude serve` and `magnitude status` show how to reach it: the address on this computer, the addresses other devices can use, and the address for apps and agents. The network access key is shown once during setup and never in logs or status. If network access is turned on in `config.json` without a key, Magnitude creates one.
  - `magnitude server setup` refuses to start if another account's Magnitude is already using port 10100, and changes nothing. If the service can't start, setup and `magnitude status` show why, including when it keeps restarting. On Ubuntu with ufw active, setup prints the command that opens the port.
  - Reinstalling the package keeps a server able to update itself, and purging the Debian or Ubuntu package removes the server's setup.
  - If you open the desktop app while the server is running, it explains that the two can't run together and lets you keep the server (and open it in your browser) or stop it and use the app, without a password on Linux.
  - `magnitude connections connect <agent>` sets up the agent for the person running the command, so connecting an agent on a server configures your own account rather than the service's.
  - `magnitude serve` on Windows keeps running when another user logs off the machine.
  - Fixes: update reports now say which version a computer updated from, and successful updates installed from the terminal are reported; installing on a Mac no longer fails when your shell profile can't be edited; a failed reinstall no longer leaves the server stopped.

- [#176](https://github.com/magnitudedev/magnitude/pull/176) [`c57fcea`](https://github.com/magnitudedev/magnitude/commit/c57fceadfd01a3580fa3c2d29b56a30b2c567f37) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - Improve grammar preparation for structured output and parallel tool calls. Large repetition bounds and shared rules no longer unnecessarily inflate certification and flattening work, and loop handling accounts for overlapping text. Fix JSON-schema additional-property keys to use valid JSON string grammar.

- [#176](https://github.com/magnitudedev/magnitude/pull/176) [`c57fcea`](https://github.com/magnitudedev/magnitude/commit/c57fceadfd01a3580fa3c2d29b56a30b2c567f37) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - - Use Magnitude from a web browser. The Magnitude service now serves the full app at `http://<computer>:10100`, the same interface as the desktop app: Discover, Catalog, My Models, Connections, Usage, Status and Settings.
  - Open the app from another device on your network. Turn on Network access in Settings, then sign in from the other device's browser with the network access key. You stay signed in across restarts and updates until you sign out or the key changes, and the app warns you before changes that would disconnect your browser (turning off network access, a new key, a different address, or a restart).
  - Run Magnitude on a headless machine with `magnitude serve` and manage it entirely from your browser: download and load models, watch usage, connect agents and change settings without the desktop app.
  - Settings that belong to the machine running Magnitude now work from every app and browser: agent connections, the model storage folder (with a folder picker inside the app that refuses folders the service can't write to), network access and machine details.
  - Updates, launch at login, restart and quit can be controlled from any connected app or browser, not only the desktop window. Controls that a headless server can't offer are hidden there.
  - If the browser loses the service, it shows "Can't reach Magnitude" and keeps retrying, so it reconnects on its own when the service comes back, however long it was gone.
  - The app works in narrow windows and on phones: below tablet width it switches to a top bar and a navigation drawer, and every page reflows to fit.
  - Removed the old unused agent web app.

## 0.2.8

### Patch Changes

- [#182](https://github.com/magnitudedev/magnitude/pull/182) [`cee4780`](https://github.com/magnitudedev/magnitude/commit/cee47802fadac4632e7d27b453650d84644777f2) Thanks [@Fazmin](https://github.com/Fazmin)! - Stop decoding a zstd Codex request once it passes the 128 MB routing limit, so a few kilobytes of compressed input can no longer expand to gigabytes of memory before being rejected.

- [`16021a5`](https://github.com/magnitudedev/magnitude/commit/16021a53aecb0efa39eed257a61b8aad0830e7bd) Thanks [@anerli](https://github.com/anerli)! - - Gemma 4 models are now paired with their assistant models in the catalog, so they use them as a draft head for faster generation.

  - Embedding tables are streamed instead of held whole in memory, so large models use less memory (Qwen3.8 27B about 600 MB less).
  - Faster Gemma decoding after the first-load optimization, and the first load stays within its time budget, so it no longer runs partly unoptimized after an upgrade.

- [`157ab94`](https://github.com/magnitudedev/magnitude/commit/157ab941334644745dc91a006470768e04a07197) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - - Fix Qwen 3.5, 3.6 and 3.8 models failing with "sampling distribution is nonfinite" on M1 Max and M1 Ultra Macs.
  - Fix 8 GB NVIDIA and AMD GPUs unloading the model when several requests run together after a long prompt. Long prompts also go much further on these GPUs (MiniCPM5 2B now handles about 78K tokens instead of failing near 26K).
  - Model recommendations and "Fits" labels on NVIDIA and AMD GPUs now match what actually loads, because memory the GPU driver keeps for itself is no longer counted as available.
  - Fix requests occasionally failing with "target rows differ" when several run at once.
  - Fix every model load crashing on AMD GPUs on Windows.
  - Fix long prompts hanging on 8 GB AMD GPUs.
  - Computers whose integrated GPU doesn't meet Magnitude's requirements, such as Intel UHD 770, now run models on the CPU instead of failing to load.

## 0.2.7

### Patch Changes

- [`3fe2c30`](https://github.com/magnitudedev/magnitude/commit/3fe2c30b6c8daa58211947df822fd67c2424fb94) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - - Magnitude is available for Arch Linux, Omarchy, and other Arch-based distributions (x64) as a pacman package. Install it with `sudo pacman -U ./magnitude-desktop.pkg.tar.zst` or the install script, and in-app updates install through pacman. As with the Debian and RPM packages, pacman refuses to upgrade or remove Magnitude while it is running.

- [`c74b8b0`](https://github.com/magnitudedev/magnitude/commit/c74b8b00b1971f910d8e5a73ae9719516363bb01) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - - Fix OpenClaw requests hanging: its `automations` tool kept the server busy without answering and could push the loaded model out of memory. These requests now answer in seconds.

  - Fix Claude Code and Oh My Pi requests failing with "Too many items" on MiniCPM5 and LFM2.5, and Oh My Pi failing on Qwen3.5 with reasoning off.
  - A response cut off by its output limit in the middle of a tool call now ends as an ordinary length stop (`length`, `max_tokens`, or `incomplete`) instead of failing with an error.
  - Fix connecting Codex from the Magnitude app on Windows, which always failed.

- [`b2f79f0`](https://github.com/magnitudedev/magnitude/commit/b2f79f00f4aa8d4ca6ad97f37a8c457ddba4e8a9) Thanks [@anerli](https://github.com/anerli)! - - Faster local inference across the board: decoding is up to about 3× faster with several requests at once, and prompt processing up to 4.5× faster (Gemma 4 E2B on Apple Silicon), on Apple Silicon and NVIDIA GPUs.

  - The one-time optimization after a download now runs in its own process within a fixed time budget, and loading a model never tunes again afterwards. A model also starts optimizing much sooner on first load (Qwen 3.8 27B: 1.9 s instead of 12.6 s).
  - Gemma 4 models use their assistant model as a draft head for faster generation.
  - Qwen3.5 4B and 9B now use their DFlash drafters for speculative decoding, so they generate considerably faster.
  - Faster prefill and decode on Apple Silicon Macs (M1 through M4): weights use a new tiled layout, prompt processing packs tokens, and attention runs on the matrix and scalar units together. The one-time optimization now plans its time across every kernel instead of searching them in order.
  - Faster time to the first token for models with a built-in drafting head: each prompt chunk is no longer computed twice.
  - Follow-up turns in a conversation reuse the previous turn's prompt even when the template renders past turns differently, so later turns start sooner.

- [`44e9fd5`](https://github.com/magnitudedev/magnitude/commit/44e9fd5d1e2fe5f5cd00c5ed5be0e0780231ec8d) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - - Fix a request without a `seed` repeating the same sample on every retry, which made some requests (such as Gemma 4 E2B over the Responses API with reasoning off) return an empty answer every time. Each request now samples with a fresh seed unless it names one, and the Responses and Anthropic APIs accept `seed` like Chat Completions.

- [`7d62661`](https://github.com/magnitudedev/magnitude/commit/7d626619f4bed1da6611028e0f27acc1e9a6f071) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - - Fix long agent sessions on Macs running the GPU out of memory, which unloaded the model mid-session. Memory used for earlier turns is now released before the GPU fills.

  - Fix NVIDIA GPUs crashing with an illegal memory access when several requests run together after a long prompt.
  - Fix large images and long prompts resetting AMD GPUs on Linux and unloading the model: GPU work is now split so no single piece runs past the driver's time limit.
  - On computers without a supported GPU, replies now stream as they are written instead of arriving all at once, stopping or abandoning a request stops it promptly instead of holding up the next one, and generation is faster on Intel and other x86 processors.
  - Fix the one-time optimization never running on Windows, which left every model on slower untuned kernels.
  - Fix Qwen3.5 4B and 9B never finishing their one-time optimization on Macs, which re-ran it on every load.
  - Fix the optimization failing on GPUs with 8 GB of memory, and on computers without a supported GPU when optimization runs again.
  - Restarting Magnitude or the computer no longer repeats the optimization on the next load.
  - The first use of a model on AMD GPUs on Linux is much faster: the optimization now prepares every kernel instead of leaving them to the first load.
  - Fix image requests failing with "insufficient memory" after a long session on AMD GPUs on Linux.
  - Fix Magnitude closing at launch when it is opened while a previous instance is still quitting, or while it takes over from `magnitude serve`.
  - `magnitude models load` no longer reports a transport error when a model takes longer than five minutes to load.
  - When the inference engine stops unexpectedly, its exit code and log now appear in Magnitude's logs.

- [`a54897b`](https://github.com/magnitudedev/magnitude/commit/a54897b2993dab95352737532531efa2468e820b) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - - Fix large images failing in vision models. An image that needed more rows than the engine's per-launch limit failed; images now encode up to 4,096 cells, and larger images are resized to fit.

- [`d066f74`](https://github.com/magnitudedev/magnitude/commit/d066f740578212b106d8316c75fcf25259256a5b) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - - Fix agent checkpoints and undo failing on Linux systems whose default file permissions are group-writable, such as Ubuntu.

  - Purging the Debian package removes Magnitude's installation state, and the login item no longer points at a removed application.
  - If an unfinished package change keeps Magnitude from starting on Linux, opening it from the application menu now shows a notice with repair steps instead of doing nothing.
  - `pacman -Qkk` reports no permission mismatch for the Arch package.
  - `nohup magnitude serve &` keeps running after you log out of an SSH session.

- [`7f65eb3`](https://github.com/magnitudedev/magnitude/commit/7f65eb322422a4679c6455b4c442ba969f2c3844) Thanks [@anerli](https://github.com/anerli)! - - macOS loads a model that fits beside wired and compressed memory instead of refusing it, including right after a download.

  - Qwen 3.8 27B loads on NVIDIA GPUs, and Gemma 4 E2B and E4B no longer hit an illegal memory access on NVIDIA GPUs.
  - The one-time optimization is more reliable: it no longer settles on slow defaults for large contexts (Gemma 4 12B at a 65,536-token context decoded at 4.8 tok/s), tunes sliding-window layers at the history they keep, and finishes in bounded time on first load.
  - Muse Glimmer can turn reasoning off.

- [`9452ff4`](https://github.com/magnitudedev/magnitude/commit/9452ff4e59517ea197597521c8fa53545b2e7626) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - - A download that stops receiving data now reports that it can't reach the model source and offers Retry, which resumes from where it stopped, instead of freezing.

  - Clicking Download right after cancelling a download now starts it again instead of doing nothing.
  - An expired or revoked `HF_TOKEN` no longer blocks downloads of public models.
  - Removing a model on Windows now deletes its files and frees the disk space.
  - Fix retrying a model download with a draft model after a network outage, which could fail again.

- [`4dba3cd`](https://github.com/magnitudedev/magnitude/commit/4dba3cd7d018660673d6b7edb303fee981bbc187) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - - Fix a stray horizontal scrollbar on Windows and Linux in windows narrower than about 1,380 pixels. Page content now fits beside the vertical scrollbar instead of extending under it.

- [`541ccf1`](https://github.com/magnitudedev/magnitude/commit/541ccf13192dab55472e1a03bb8a9a80a9d20cda) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - - Magnitude writes logs again: `service.log`, `inference.log` and `desktop.log` in `~/.magnitude/logs` record the service, the inference engine and the app, including crashes and restarts. Each file is size-capped. Attach them when reporting a problem.

## 0.2.6

### Patch Changes

- [`02568d6`](https://github.com/magnitudedev/magnitude/commit/02568d6dd2ec52edd8b521e49d8571301f87a5ab) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - - Fix Codex failing to send tool results back to a local model. Reasoning items that a client replays with `content` or `summary` set to null are now accepted as empty.

- [`835a476`](https://github.com/magnitudedev/magnitude/commit/835a4761bca9b2bbc619e11c14586c49d3e1bbdc) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - - Fix an OpenCode session being rejected with "assistant content is required unless tool_calls are present" after a step that failed before producing any output. An empty assistant turn in the history is now skipped, in both the Chat Completions and Anthropic APIs.

- [`2a481f7`](https://github.com/magnitudedev/magnitude/commit/2a481f7c5d3c78e8e16ddf4f551ba0c13ba9c16f) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - - Fix a headless `magnitude serve` that failed to start crashing with EBADF instead of reporting the error that stopped it.

- [`76d35d7`](https://github.com/magnitudedev/magnitude/commit/76d35d737355f0f5dc3aad3c36ec7ecf5611cfa2) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - - Fix long requests to a local model failing with a 502 after about five minutes. A non-streaming generation or a long prompt sends nothing until it finishes, and the connection to the engine no longer times out while it waits.

- [`d5bf92d`](https://github.com/magnitudedev/magnitude/commit/d5bf92d087805b6d78cebe8a918fcece824fd20d) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - - Fix models failing on M1 and M2 Macs during long prompts with "device lost: … Impacting Interactivity", after which the model stayed unloaded. The engine now relaxes the macOS GPU watchdog at start (as llama.cpp does), so prompts of 35k and 69k tokens on Gemma 4 26B complete instead of failing at about 20k.

- [`01a1728`](https://github.com/magnitudedev/magnitude/commit/01a17287bd4dfb15b61e0861a8f17a6db12eadcf) Thanks [@anerli](https://github.com/anerli)! - - Speed up prompt processing on M5 and later Macs by about 50%: Qwen3.5-4B at a 64K context now processes prompts at about 970 tok/s (previously 652). Attention over the prompt reads keys and values directly through the GPU's tensor operations instead of staging them, and kernel tuning no longer keeps a slower default whose own timing was unstable.

  - Speed up prompt processing on every Mac by decoding each block of weights once for up to 512 rows instead of once per 64: matrix multiplies run 7–11% faster on an M4 Pro and 17–25% faster on an M1, with identical output. Qwen3.5-4B at a 64K context processes prompts at 534 tok/s on an M4 Pro (previously 513).

- [#166](https://github.com/magnitudedev/magnitude/pull/166) [`c738ead`](https://github.com/magnitudedev/magnitude/commit/c738ead74d6176861311ac43567eb36d8023d8c7) Thanks [@aaronjensen](https://github.com/aaronjensen)! - - Fix models failing to load on some Macs (for example Qwen 3.6 on an M5 Max) with a Metal shader compilation error such as "no template named 'extents' in namespace 'metal'". Metal kernels are now compiled with the same language version the device was probed with, so kernels that use tensor operations build wherever the probe found them available.

- [`44af293`](https://github.com/magnitudedev/magnitude/commit/44af293bd9968ba0e428437bd5bdc6f5274218a4) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - - Fix Oh My Pi and OpenClaw failing on Qwen models when their tools take free-form JSON among optional properties. Where a model's grammar for parallel tool calls cannot be compiled efficiently but its grammar for a single call can, the request now allows one tool call per turn.

- [`048a92d`](https://github.com/magnitudedev/magnitude/commit/048a92de8c5fc971250207cb5dea7890a8b39ccf) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - - Fix the Windows engine aborting when a chat template or tool call produced invalid JSON. The template library is now built with C++ exception handling on MSVC, so JSON errors are reported instead of crashing the engine.

## 0.2.5

### Patch Changes

- [`32e095e`](https://github.com/magnitudedev/magnitude/commit/32e095e6561b2e58daaaf1438604a21d36ee827f) Thanks [@anerli](https://github.com/anerli)! - - Speed up generation for mixture-of-experts models with multi-token prediction by drafting three tokens ahead instead of one: Qwen3.6-35B-A3B now generates 106–138 tok/s on a GB10 (previously 97–105) and 109–138 tok/s on an M4 Pro (previously 103–111).

  - Speed up generation at a 16K context by about 8% on Macs (67.0 → 72.4 tok/s) and 11% on NVIDIA GPUs (66.3 → 73.4 tok/s), with the same output, by choosing each token while reading less of the output layer and doing more of each step in fewer GPU launches.
  - Speed up multi-token prediction on NVIDIA GPUs by loading each expert's weights once per step when several drafted tokens choose it, cutting verification time by up to 11%.

- [`32e095e`](https://github.com/magnitudedev/magnitude/commit/32e095e6561b2e58daaaf1438604a21d36ee827f) Thanks [@anerli](https://github.com/anerli)! - - Speed up prompt processing on M5 and later Macs about 2x by running matrix multiplies and attention on the GPU's tensor operations: Qwen3.5-4B at a 64K context now processes prompts at 649 tok/s (previously 308), cutting time to first token from 213 to 101 seconds, with identical output. Other Macs are unchanged.

- [`7aea836`](https://github.com/magnitudedev/magnitude/commit/7aea83653a5acf2035d6d3229efd5700ff684e8d) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - - Fix models failing to load on M1 and M2 Macs with "requests N threads per threadgroup; the pipeline allows M". Metal kernels are now built to accept the thread count they launch with, which fixes Qwen3.8 27B on M1 Max and similar errors in other kernels.

  - Fix models with 16 or more query heads per key (Gemma 4 12B, Muse Glimmer 30B, Nemotron 3.5 Lightning, Qwen3.5 122B, Nemotron 3 Super) failing on M1 and M2 Macs. Prefill attention now splits a key's query heads into groups, so it fits every Mac's thread limit, with no change in speed or output elsewhere.

- [`cffe46e`](https://github.com/magnitudedev/magnitude/commit/cffe46e77af5f45ce99565f096d545443dbbd3d0) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - - Fix removing a model that is running or loading failing with a misleading error. Removing it now stops the model first, and the confirmation says so.

- [`a3e5422`](https://github.com/magnitudedev/magnitude/commit/a3e542241b075567894881290aad2a5ed2f94aaf) Thanks [@anerli](https://github.com/anerli)! - - Fix models with DFlash2 speculative decoding (Qwen3.8 27B) and Nemotron models failing to load on Vulkan GPUs with a shader compilation error. Every GPU kernel is now compiled for Vulkan, CUDA and Metal before each release, including every kernel each catalog model loads.

## 0.2.4

### Patch Changes

- [`f0498ce`](https://github.com/magnitudedev/magnitude/commit/f0498ce285e4e97815ba16dd74044ce5efebb63a) Thanks [@anerli](https://github.com/anerli)! - - Keep first-load kernel tuning within its minute for large models too, such as Gemma 4 26B: preparing each kernel's test data now counts against the same time, is done once instead of twice, and is skipped for kernels whose share of the minute cannot cover it, which keep their default configuration.

- [`e38af0e`](https://github.com/magnitudedev/magnitude/commit/e38af0e9a232293dca12c5bad3a91b96e58ca12e) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - - Fix models failing to load on M1 and M2 Macs when a kernel's default configuration needs more threads than the chip allows for it. Tuning now starts from the nearest configuration that runs.

- [`cef7f3b`](https://github.com/magnitudedev/magnitude/commit/cef7f3bb9c2dd06d96aeb59dde8039b6f83386a5) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - - Fix requests that fail partway through, for example when memory runs short during a long conversation, returning an empty reply that looked like success. They now return a 503 with `Retry-After` and a message agent harnesses recognize, so Pi, OpenCode, Claude Code and others retry them automatically.

  - Log memory pressure and request failures from the inference engine, which previously failed silently.

- [`4713d97`](https://github.com/magnitudedev/magnitude/commit/4713d97b4b913a3ed2194f53826ae01658600839) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - - Fix `magnitude serve` failing to start on a Mac reached over SSH with nobody logged in at the console.

- [`7eabf99`](https://github.com/magnitudedev/magnitude/commit/7eabf993ebbd96a4ab2c067d2c7f2e7d3e4ff8f8) Thanks [@anerli](https://github.com/anerli)! - - Reduce the memory a model needs beyond its weights, so larger models and longer contexts fit: Qwen3.5-4B's working memory fell from 3.7 GB to 120 MB, and the memory reserved to run it from 10.0 GB to 5.9 GB, at the same speed. Image-processing memory is now claimed on the first image and released when idle.

- [`7eabf99`](https://github.com/magnitudedev/magnitude/commit/7eabf993ebbd96a4ab2c067d2c7f2e7d3e4ff8f8) Thanks [@anerli](https://github.com/anerli)! - - Fix the first load of a model taking 20–50 minutes and appearing to hang on CPU-only machines. Kernel tuning now takes at most about a minute on any device, instead of a fixed number of configurations whose time grew with the device's slowness, and its progress reports that minute. On an M4 Max, a first load of Qwen3.5-4B now tunes in about 50 seconds (previously 149 on the GPU and 279 on the CPU) with the same speed afterwards.
  - Speed up CPU inference on x86 for models with Q6_K weights by converting their half-precision scales without a slow processor path.

## 0.2.3

### Patch Changes

- [`5b1aba8`](https://github.com/magnitudedev/magnitude/commit/5b1aba81185ac8b07ba363ba2a11427c3447d62a) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - - Fix the Windows installer failing on Windows 10 with "An interrupted installation could not be recovered (code 4395)". Setup now installs normally and publishes the `magnitude` command to PATH, which also failed on Windows 10 once installation got past that error.

## 0.2.2

### Patch Changes

- [`1588451`](https://github.com/magnitudedev/magnitude/commit/1588451a88f979a70c1f6385599d50de029064f7) Thanks [@anerli](https://github.com/anerli)! - - Fix the app getting stuck on "Assessing models" on some hardware: model speed is now estimated from the device's memory bandwidth instead of running kernels on the GPU, which could hang or fail.

- [`3940408`](https://github.com/magnitudedev/magnitude/commit/394040878bd6cf10bf5a9addd9a3791d8eb2ef9f) Thanks [@anerli](https://github.com/anerli)! - - Fix Codex hanging after its first tool call over the Responses WebSocket: follow-up requests now continue from the previous response's output, and request errors end the request instead of leaving it waiting.
  - Fix requests that repeat the same image, in one message or across turns, failing with a 400. A repeated image is now encoded once.
  - Fix tool call IDs repeating across turns (every turn's first call was `call_0`), which made Claude Code drop tool calls and loop.
  - Fix Anthropic token usage counting cached tokens twice in responses and reporting none when streaming, and `count_tokens` requiring `max_tokens`.
  - Fix large system prompts being re-read in full when only the last message changes: later requests now resume from the cached prompt.
  - Fix tools with free-form object parameters failing on Gemma 4 with "Too many items" (breaking Claude Code and Oh My Pi), Cline failing mid-task with "Output parser would retract a published tool call", and forced tool calls (`tool_choice` "required", "any" or a named tool) repeating until the token limit.

## 0.2.1

### Patch Changes

- [`0476b71`](https://github.com/magnitudedev/magnitude/commit/0476b71a0fb772245328529312a3392d77b630aa) Thanks [@anerli](https://github.com/anerli)! - - Fix the local model hanging forever when a request arrived while another was generating, as with Qwen3.6 35B-A3B: every later request waited without a response while the model still reported Ready, and the engine held a CPU core at 100%. Admission no longer waits on the running generation, and the engine can no longer wait on work only it could release.
  - Fix speculative-decoding models failing mid-request with "only a blocked last page is relocated" during long prompts.

## 0.2.0

### Minor Changes

- [`f0cd67e`](https://github.com/magnitudedev/magnitude/commit/f0cd67ed900fe76022273081490be0939908df70) Thanks [@anerli](https://github.com/anerli)! - Replace the llama.cpp-based inference engine with Magnitude's own engine which automatically optimizes itself for any hardware and has efficient kernels for several open-weight model families.

### Patch Changes

- [`9b929cf`](https://github.com/magnitudedev/magnitude/commit/9b929cfef434742e0f43043dfbf212f7801d5ba5) Thanks [@anerli](https://github.com/anerli)! - - Fix models that load and run, such as Gemma 4 26B-A4B and Qwen3.6 35B-A3B on Apple Silicon, being reported as unable to run on this computer.

  - A model is reported as unsupported only when Magnitude cannot actually run it. A gap in the device's speed measurements now shows "Speed estimate unavailable" instead of hiding the model.

- [`482e6ab`](https://github.com/magnitudedev/magnitude/commit/482e6abbb94f4081c19e223fef09263ac559df14) Thanks [@anerli](https://github.com/anerli)! - - Make DFlash, DSpark, and DFlash2 speculative decoding faster than plain decoding on Apple Silicon (Qwen3.6-35B-A3B at 65k tokens: 10.7% faster, previously 7% slower) and faster on NVIDIA (38.8% over plain, previously 28.7%), with better draft acceptance at long context.

  - Fix speculative drafts whose layers are all sliding-window (such as Muse-Glimmer's DFlash) failing to load.
  - Reduce the time to first token added by speculative decoding from about 2.6% to 0.6% of prompt processing on NVIDIA and from about 1.5% to 0.7% on Apple Silicon.
  - Speed up long-context decoding and speculative verification by reading each attention head group's history once, on Apple Silicon, NVIDIA, and Vulkan GPUs.
  - Speed up mixture-of-experts decoding on Apple Silicon and prompt processing on NVIDIA.

- [`07c39a1`](https://github.com/magnitudedev/magnitude/commit/07c39a14870d8691d7762a3601b913b140ba3319) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - - Add `magnitude serve` to run inference without a desktop window on macOS, Windows, and Linux. Opening Desktop takes over from the foreground server and reports why it stopped.

  - Replace the `magnitude service` commands with `magnitude serve` and `magnitude status`. Model, catalog, hardware, and connection commands require an existing Desktop or server instead of starting one automatically. Startup errors identify which application must be stopped.
  - Share application updates between Desktop and the CLI. A running server can prepare updates without being interrupted; prepared updates install at the next startup. The `magnitude update` commands support checking, downloading, inspecting, installing, and discarding updates while Desktop is closed. Failed installations require an explicit retry.
  - Fix Windows update preparation when the update folder has inherited permissions, and improve update recovery and command continuation. Existing affected releases still require a manual installer to receive the fix.
  - Add shell and PowerShell installation scripts for the complete application, including its CLI.
  - Improve `magnitude app open` during Desktop takeover. On Windows, clicking the tray icon opens Desktop, and sharper tray icons adapt to the system's light or dark theme.
  - Add a remote server guide and update network access, CLI, and installation documentation.

- [`f29bcb2`](https://github.com/magnitudedev/magnitude/commit/f29bcb218cc60d6cf01c930a54cd6e735ba4ba7f) Thanks [@anerli](https://github.com/anerli)! - - Fix long prompts failing partway through with "target graph class ... was not sealed" and the model server going down, as with Qwen3.8-27B on a 64k-token prompt on Apple Silicon. Attention history now stays within the bound its kernels were prepared for on every model, however requests interleave, fork or are reclaimed.

- [#125](https://github.com/magnitudedev/magnitude/pull/125) [`fd37123`](https://github.com/magnitudedev/magnitude/commit/fd37123a374ee4931a0f2bb28e6b3cb091a6d601) Thanks [@Nitish-1303](https://github.com/Nitish-1303)! - Gate remote callers in the /rpc and inference route handlers instead of the middleware, so case, slash, and percent-encoded path variants can no longer skip the API key check.

## 0.2.0-alpha.2

### Patch Changes

- [`9b929cf`](https://github.com/magnitudedev/magnitude/commit/9b929cfef434742e0f43043dfbf212f7801d5ba5) Thanks [@anerli](https://github.com/anerli)! - - Fix models that load and run, such as Gemma 4 26B-A4B and Qwen3.6 35B-A3B on Apple Silicon, being reported as unable to run on this computer.

  - A model is reported as unsupported only when Magnitude cannot actually run it. A gap in the device's speed measurements now shows "Speed estimate unavailable" instead of hiding the model.

- [`f29bcb2`](https://github.com/magnitudedev/magnitude/commit/f29bcb218cc60d6cf01c930a54cd6e735ba4ba7f) Thanks [@anerli](https://github.com/anerli)! - - Fix long prompts failing partway through with "target graph class ... was not sealed" and the model server going down, as with Qwen3.8-27B on a 64k-token prompt on Apple Silicon. Attention history now stays within the bound its kernels were prepared for on every model, however requests interleave, fork or are reclaimed.

## 0.2.0-alpha.1

### Patch Changes

- [`482e6ab`](https://github.com/magnitudedev/magnitude/commit/482e6abbb94f4081c19e223fef09263ac559df14) Thanks [@anerli](https://github.com/anerli)! - - Make DFlash, DSpark, and DFlash2 speculative decoding faster than plain decoding on Apple Silicon (Qwen3.6-35B-A3B at 65k tokens: 10.7% faster, previously 7% slower) and faster on NVIDIA (38.8% over plain, previously 28.7%), with better draft acceptance at long context.
  - Fix speculative drafts whose layers are all sliding-window (such as Muse-Glimmer's DFlash) failing to load.
  - Reduce the time to first token added by speculative decoding from about 2.6% to 0.6% of prompt processing on NVIDIA and from about 1.5% to 0.7% on Apple Silicon.
  - Speed up long-context decoding and speculative verification by reading each attention head group's history once, on Apple Silicon, NVIDIA, and Vulkan GPUs.
  - Speed up mixture-of-experts decoding on Apple Silicon and prompt processing on NVIDIA.

## 0.2.0-alpha.0

### Minor Changes

- [`f0cd67e`](https://github.com/magnitudedev/magnitude/commit/f0cd67ed900fe76022273081490be0939908df70) Thanks [@anerli](https://github.com/anerli)! - Replace the llama.cpp-based inference engine with Magnitude's own engine which automatically optimizes itself for any hardware and has efficient kernels for several open-weight model families.

### Patch Changes

- [`07c39a1`](https://github.com/magnitudedev/magnitude/commit/07c39a14870d8691d7762a3601b913b140ba3319) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - - Add `magnitude serve` to run inference without a desktop window on macOS, Windows, and Linux. Opening Desktop takes over from the foreground server and reports why it stopped.

  - Replace the `magnitude service` commands with `magnitude serve` and `magnitude status`. Model, catalog, hardware, and connection commands require an existing Desktop or server instead of starting one automatically. Startup errors identify which application must be stopped.
  - Share application updates between Desktop and the CLI. A running server can prepare updates without being interrupted; prepared updates install at the next startup. The `magnitude update` commands support checking, downloading, inspecting, installing, and discarding updates while Desktop is closed. Failed installations require an explicit retry.
  - Fix Windows update preparation when the update folder has inherited permissions, and improve update recovery and command continuation. Existing affected releases still require a manual installer to receive the fix.
  - Add shell and PowerShell installation scripts for the complete application, including its CLI.
  - Improve `magnitude app open` during Desktop takeover. On Windows, clicking the tray icon opens Desktop, and sharper tray icons adapt to the system's light or dark theme.
  - Add a remote server guide and update network access, CLI, and installation documentation.

- [`fd37123`](https://github.com/magnitudedev/magnitude/commit/fd37123a374ee4931a0f2bb28e6b3cb091a6d601) - Gate remote callers in the /rpc and inference route handlers instead of the middleware, so case, slash, and percent-encoded path variants can no longer skip the API key check.

## 0.1.5

### Patch Changes

- [`d157b35`](https://github.com/magnitudedev/magnitude/commit/d157b35a79f41ebd3d8b0017d7f2ee0471173a62) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - Clamp reported available system memory to physical capacity so model loading does not fail on macOS memory samples that briefly exceed installed RAM, stop an inherited `ELECTRON_RUN_AS_NODE` (for example from a VS Code terminal) from breaking desktop app launch, and size the macOS app icon to Apple's icon grid so it matches other Dock icons.

## 0.1.4

### Patch Changes

- [`6b56b68`](https://github.com/magnitudedev/magnitude/commit/6b56b68640051e65b4a1e28e053e3b11d74518b0) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - Add a configurable model storage folder, opt-in network access with a generated API key so other devices, containers, and WSL can use local inference, a redesigned Settings page, the real version in development builds, and tighter CORS and WebSocket origin checks on the local service.

## 0.1.3

### Patch Changes

- [#113](https://github.com/magnitudedev/magnitude/pull/113) [`8440631`](https://github.com/magnitudedev/magnitude/commit/84406317e6df9a62ad54e0e0dc117e7898ca034f) Thanks [@lepsistemas](https://github.com/lepsistemas)! - Handle native driver output preceding the backend eligibility JSON record.

## 0.1.2

### Patch Changes

- [`5bcb95b`](https://github.com/magnitudedev/magnitude/commit/5bcb95b2b6449ca5e74b3d28b6b22cb2a52549a3) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - Add model-specific copyable agent commands with compact model selectors, guide downloaded Discover recommendations to Connections, and improve dropdown and copy feedback.

## 0.1.1

### Patch Changes

- [`e216361`](https://github.com/magnitudedev/magnitude/commit/e216361bffd4b0d53459299b1caa7f61e571cdb5) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - Smoothly upgrade previous standalone macOS and Linux installations while preserving downloaded models. Fix macOS window dragging and automatically register fresh installed macOS apps for launch at login while preserving later opt-outs. Simplify connection refresh controls and keep recommendation names on one line with quantization visible.

## 0.1.0

### Minor Changes

- [#105](https://github.com/magnitudedev/magnitude/pull/105) [`9d52ca8`](https://github.com/magnitudedev/magnitude/commit/9d52ca81d18fc98a26f81c7a8e7b40c064392ca6) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - Move local-model onboarding into the Magnitude desktop application. The bundled CLI is headless and no longer hosts interactive onboarding or the Magnitude harness. Desktop installation exposes the magnitude command without npm. Pi connections configure the external harness without installing the Magnitude Pi extension.

## 0.1.0-alpha.0

### Minor Changes

- [#105](https://github.com/magnitudedev/magnitude/pull/105) [`9d52ca8`](https://github.com/magnitudedev/magnitude/commit/9d52ca81d18fc98a26f81c7a8e7b40c064392ca6) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - Move local-model onboarding into the Magnitude desktop application. The bundled CLI is headless and no longer hosts interactive onboarding or the Magnitude harness. Desktop installation exposes the magnitude command without npm. Pi connections configure the external harness without installing the Magnitude Pi extension.

## 0.0.15

### Patch Changes

- [`a5a5711`](https://github.com/magnitudedev/magnitude/commit/a5a5711a7d7c0deee96f3c476df4711bc1df409c) Thanks [@anerli](https://github.com/anerli)! - Add AssociatedBundleIdentifiers to launchd plist for macOS so login item shows correct icon and name

- [`d349ed8`](https://github.com/magnitudedev/magnitude/commit/d349ed8d06308f140393270eab7168fe793b38fd) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - Show a desktop migration notice during installation and setup.

- [`1cbaafc`](https://github.com/magnitudedev/magnitude/commit/1cbaafc04ad38393c7c54b1ae4ea70dc5ab4c32f) Thanks [@anerli](https://github.com/anerli)! - Apple signing

## 0.0.15-alpha.1

### Patch Changes

- [`a5a5711`](https://github.com/magnitudedev/magnitude/commit/a5a5711a7d7c0deee96f3c476df4711bc1df409c) Thanks [@anerli](https://github.com/anerli)! - Add AssociatedBundleIdentifiers to launchd plist for macOS so login item shows correct icon and name

## 0.0.15-alpha.0

### Patch Changes

- [`1cbaafc`](https://github.com/magnitudedev/magnitude/commit/1cbaafc04ad38393c7c54b1ae4ea70dc5ab4c32f) Thanks [@anerli](https://github.com/anerli)! - Apple signing

## 0.0.14

### Patch Changes

- [`cfe32b7`](https://github.com/magnitudedev/magnitude/commit/cfe32b74c9cb03fc428e5a03abaee39899749c53) Thanks [@anerli](https://github.com/anerli)! - update pi plugin UI

## 0.0.13

### Patch Changes

- [`d7ad43d`](https://github.com/magnitudedev/magnitude/commit/d7ad43df99763a7460f0e0449ad9cab606ca6648) Thanks [@anerli](https://github.com/anerli)! - fix: use Standard instead of Background process type for macOS service to prevent throttling

## 0.0.12

### Patch Changes

- [`183f48c`](https://github.com/magnitudedev/magnitude/commit/183f48c9a7b146549ce24c9d52e1ea171376f176) Thanks [@anerli](https://github.com/anerli)! - Fix macOS service detection under non-C locales by making process-start identity formatting consistent with the background service.

- [`6b894d9`](https://github.com/magnitudedev/magnitude/commit/6b894d93c709b98e11470356ebb3030401fbdfdf) Thanks [@anerli](https://github.com/anerli)! - fix pi package pre tag parsing

- [`6fbb57f`](https://github.com/magnitudedev/magnitude/commit/6fbb57fed6c1c0e833a0109a24b3485773899f9d) Thanks [@anerli](https://github.com/anerli)! - simplify pi onboard logic

- [`d7a26c9`](https://github.com/magnitudedev/magnitude/commit/d7a26c9aa3a6fe72a950e727512f9ab11937e744) Thanks [@anerli](https://github.com/anerli)! - pi onboard

- [#71](https://github.com/magnitudedev/magnitude/pull/71) [`afe6288`](https://github.com/magnitudedev/magnitude/commit/afe6288841ace31b4a4f974003ce9e9f0430ae62) Thanks [@lloydgreenwald](https://github.com/lloydgreenwald)! - Set an explicit HTTP request-body limit so larger image requests are no longer rejected before reaching the handler.

- [`dc36e21`](https://github.com/magnitudedev/magnitude/commit/dc36e2191652ae8a8eea35bddf8ef6ad32f76aa7) Thanks [@anerli](https://github.com/anerli)! - fix: assume rpc revision version 0 when not included in health, also try start cli from sdk once if version mismatch

## 0.0.12-alpha.4

### Patch Changes

- [`6b894d9`](https://github.com/magnitudedev/magnitude/commit/6b894d93c709b98e11470356ebb3030401fbdfdf) Thanks [@anerli](https://github.com/anerli)! - fix pi package pre tag parsing

## 0.0.12-alpha.3

### Patch Changes

- [`6fbb57f`](https://github.com/magnitudedev/magnitude/commit/6fbb57fed6c1c0e833a0109a24b3485773899f9d) Thanks [@anerli](https://github.com/anerli)! - simplify pi onboard logic

## 0.0.12-alpha.2

### Patch Changes

- [`d7a26c9`](https://github.com/magnitudedev/magnitude/commit/d7a26c9aa3a6fe72a950e727512f9ab11937e744) Thanks [@anerli](https://github.com/anerli)! - pi onboard

## 0.0.12-alpha.1

### Patch Changes

- [`183f48c`](https://github.com/magnitudedev/magnitude/commit/183f48c9a7b146549ce24c9d52e1ea171376f176) Thanks [@anerli](https://github.com/anerli)! - Fix macOS service detection under non-C locales by making process-start identity formatting consistent with the background service.

- [`dc36e21`](https://github.com/magnitudedev/magnitude/commit/dc36e2191652ae8a8eea35bddf8ef6ad32f76aa7) Thanks [@anerli](https://github.com/anerli)! - fix: assume rpc revision version 0 when not included in health, also try start cli from sdk once if version mismatch

## 0.0.12-alpha.0

### Patch Changes

- [#71](https://github.com/magnitudedev/magnitude/pull/71) [`afe6288`](https://github.com/magnitudedev/magnitude/commit/afe6288841ace31b4a4f974003ce9e9f0430ae62) Thanks [@lloydgreenwald](https://github.com/lloydgreenwald)! - Set an explicit HTTP request-body limit so larger image requests are no longer rejected before reaching the handler.

## 0.0.11

### Patch Changes

- [`a18e40a`](https://github.com/magnitudedev/magnitude/commit/a18e40a26c75ae30e005ea6b2a87d70208bfcebf) Thanks [@anerli](https://github.com/anerli)! - More robust service health checks

- [`a18e40a`](https://github.com/magnitudedev/magnitude/commit/a18e40a26c75ae30e005ea6b2a87d70208bfcebf) Thanks [@anerli](https://github.com/anerli)! - Better inference errors for context length exceeded

- [`a18e40a`](https://github.com/magnitudedev/magnitude/commit/a18e40a26c75ae30e005ea6b2a87d70208bfcebf) Thanks [@anerli](https://github.com/anerli)! - Improve agent onboarding doc

## 0.0.10

### Patch Changes

- [`88a9987`](https://github.com/magnitudedev/magnitude/commit/88a9987267c58ed1633ffa58a896dd3c55dfb6e0) Thanks [@anerli](https://github.com/anerli)! - fix: template inspections failing after model download through CLI

- [`88a9987`](https://github.com/magnitudedev/magnitude/commit/88a9987267c58ed1633ffa58a896dd3c55dfb6e0) Thanks [@anerli](https://github.com/anerli)! - fix: CLI commands hanging after completion

## 0.0.9

### Patch Changes

- [`7f0ef1b`](https://github.com/magnitudedev/magnitude/commit/7f0ef1bcc83fc80e2b606b7f48aee85ed7316457) Thanks [@anerli](https://github.com/anerli)! - Load heavy CLI command runtimes only when their command is selected, reducing version and help startup overhead.

- [`7759adf`](https://github.com/magnitudedev/magnitude/commit/7759adf540c22141597bd4b7743db6fe310533c8) Thanks [@anerli](https://github.com/anerli)! - Better headless CLI interface

## 0.0.9-alpha.0

### Patch Changes

- [`7f0ef1b`](https://github.com/magnitudedev/magnitude/commit/7f0ef1bcc83fc80e2b606b7f48aee85ed7316457) Thanks [@anerli](https://github.com/anerli)! - Load heavy CLI command runtimes only when their command is selected, reducing version and help startup overhead.

## 0.0.8

### Patch Changes

- [`0c82137`](https://github.com/magnitudedev/magnitude/commit/0c82137eb9a4d5042a42accfb3235b0505e479fd) Thanks [@anerli](https://github.com/anerli)! - Add Qwen 3.8 Flash Next to catalog

- [`5e46594`](https://github.com/magnitudedev/magnitude/commit/5e4659408029d20ec056fad8c3967a37c553c2fb) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - Improve onboarding model selection, responsive layout, and mouse interactions

- [`a35372a`](https://github.com/magnitudedev/magnitude/commit/a35372aeadac5fd6e0ef1bd574116fe504dab3b8) Thanks [@anerli](https://github.com/anerli)! - DSpark support for LFM 2.5 models

- [`1af97f8`](https://github.com/magnitudedev/magnitude/commit/1af97f8df0110295bec46acdc652ed18cc1b05e3) Thanks [@anerli](https://github.com/anerli)! - move setup from --setup flag to setup subcommand

- [`2f4ce23`](https://github.com/magnitudedev/magnitude/commit/2f4ce23de9fc75a82b652ff60eb96f2fb2508585) Thanks [@anerli](https://github.com/anerli)! - Connect inference to other harnesses, direct CLI controls, more intuitive onboarding

## 0.0.8-alpha.3

### Patch Changes

- [`a35372a`](https://github.com/magnitudedev/magnitude/commit/a35372aeadac5fd6e0ef1bd574116fe504dab3b8) Thanks [@anerli](https://github.com/anerli)! - DSpark support for LFM 2.5 models

## 0.0.8-alpha.2

### Patch Changes

- [`1af97f8`](https://github.com/magnitudedev/magnitude/commit/1af97f8df0110295bec46acdc652ed18cc1b05e3) Thanks [@anerli](https://github.com/anerli)! - move setup from --setup flag to setup subcommand

## 0.0.8-alpha.1

### Patch Changes

- [`5e46594`](https://github.com/magnitudedev/magnitude/commit/5e4659408029d20ec056fad8c3967a37c553c2fb) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - Improve onboarding model selection, responsive layout, and mouse interactions

## 0.0.8-alpha.0

### Patch Changes

- [`2f4ce23`](https://github.com/magnitudedev/magnitude/commit/2f4ce23de9fc75a82b652ff60eb96f2fb2508585) Thanks [@anerli](https://github.com/anerli)! - Connect inference to other harnesses, direct CLI controls, more intuitive onboarding

## 0.0.7

### Patch Changes

- [`413fbdf`](https://github.com/magnitudedev/magnitude/commit/413fbdfd1f73dfec53fe87e89626a3e6034faef3) Thanks [@anerli](https://github.com/anerli)! - auto-update system

- [`e8d8204`](https://github.com/magnitudedev/magnitude/commit/e8d82048aad277e37e7eb5739f68cfe095fbdf4c) Thanks [@anerli](https://github.com/anerli)! - fix: race condition in rpc finalization sometimes causing defect on auto-update

- [`3414375`](https://github.com/magnitudedev/magnitude/commit/3414375095513a1ac5a2bcef0ce543bcd00e0720) Thanks [@anerli](https://github.com/anerli)! - fix: daemon process coordination issues

- [`f6e1a09`](https://github.com/magnitudedev/magnitude/commit/f6e1a090dbc8a46daed20e8e2f6b008d73a92532) Thanks [@anerli](https://github.com/anerli)! - Update Qwen 3.8 27B to use Unsloth Dynamic V3.0 GGUFs

## 0.0.7-alpha.3

### Patch Changes

- [`3414375`](https://github.com/magnitudedev/magnitude/commit/3414375095513a1ac5a2bcef0ce543bcd00e0720) Thanks [@anerli](https://github.com/anerli)! - fix: daemon process coordination issues

## 0.0.7-alpha.2

### Patch Changes

- [`e8d8204`](https://github.com/magnitudedev/magnitude/commit/e8d82048aad277e37e7eb5739f68cfe095fbdf4c) Thanks [@anerli](https://github.com/anerli)! - fix: race condition in rpc finalization sometimes causing defect on auto-update

## 0.0.7-alpha.1

### Patch Changes

- [`f6e1a09`](https://github.com/magnitudedev/magnitude/commit/f6e1a090dbc8a46daed20e8e2f6b008d73a92532) Thanks [@anerli](https://github.com/anerli)! - Update Qwen 3.8 27B to use Unsloth Dynamic V3.0 GGUFs

## 0.0.7-alpha.0

### Patch Changes

- [`413fbdf`](https://github.com/magnitudedev/magnitude/commit/413fbdfd1f73dfec53fe87e89626a3e6034faef3) Thanks [@anerli](https://github.com/anerli)! - auto-update system

## 0.0.6

### Patch Changes

- [`f9eb747`](https://github.com/magnitudedev/magnitude/commit/f9eb747e260acc35cd5695f779c067911bc0f7e7) Thanks [@anerli](https://github.com/anerli)! - better onboarding view, better model catalog view

- [`f9eb747`](https://github.com/magnitudedev/magnitude/commit/f9eb747e260acc35cd5695f779c067911bc0f7e7) Thanks [@anerli](https://github.com/anerli)! - fix: support multimodal projectors properly and fix multimodal interactions with dflash/dspark

## 0.0.5

### Patch Changes

- [`e23e3f2`](https://github.com/magnitudedev/magnitude/commit/e23e3f2c5e42e86b348cb91b79cd30036721ed36) Thanks [@anerli](https://github.com/anerli)! - update drafters for deepseek v4 flash, nemotron lightning, muse glimmer

- [`d319b12`](https://github.com/magnitudedev/magnitude/commit/d319b1221b39f4f9e36521367721e75b9a11c840) Thanks [@anerli](https://github.com/anerli)! - add qwen 3.8 27b support

## 0.0.4

### Patch Changes

- [`228d2c3`](https://github.com/magnitudedev/magnitude/commit/228d2c33a9ed60e3fc4bc163d84b043baed1d252) Thanks [@anerli](https://github.com/anerli)! - feat: dflash and dspark support

- [`228d2c3`](https://github.com/magnitudedev/magnitude/commit/228d2c33a9ed60e3fc4bc163d84b043baed1d252) Thanks [@anerli](https://github.com/anerli)! - fix: simplify stored installations

## 0.0.3

### Patch Changes

- [`a3df81f`](https://github.com/magnitudedev/magnitude/commit/a3df81f0b4098572dc595abdb620b620b019e3fb) Thanks [@anerli](https://github.com/anerli)! - fix: preserve prompt cache on interrupt

- [`f368a70`](https://github.com/magnitudedev/magnitude/commit/f368a70dd879a185b85688a71e28294b83067600) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - Add support for Nvidia Nemotron 3.5 Lightning and NVFP4

- [#28](https://github.com/magnitudedev/magnitude/pull/28) [`f9d692c`](https://github.com/magnitudedev/magnitude/commit/f9d692c5c0d431002a6158cfa63461771348517e) Thanks [@fabianhug](https://github.com/fabianhug)! - Exit with WSL guidance instead of failing to download artifacts when the launcher runs on native Windows.

## 0.0.2

### Patch Changes

- [`5635b0a`](https://github.com/magnitudedev/magnitude/commit/5635b0a667fab490974e11aebfae768167e8da74) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - Add Muse Glimmer 30B to catalog, update llama.cpp version

## 0.0.1

### Patch Changes

- [`ba17e6f`](https://github.com/magnitudedev/magnitude/commit/ba17e6f967fc9ed3eaa991d8a8da547cd6867e3f) Thanks [@anerli](https://github.com/anerli)! - rm advisor

- [`d5489f6`](https://github.com/magnitudedev/magnitude/commit/d5489f6ea193a3d8b30173e2f8b3a06104c470bc) Thanks [@anerli](https://github.com/anerli)! - feat: configurable custom chat completions endpoints

- [#19](https://github.com/magnitudedev/magnitude/pull/19) [`fcdd491`](https://github.com/magnitudedev/magnitude/commit/fcdd491ad6e813feb47601b084b4187645be123a) Thanks [@nicolasdmolina](https://github.com/nicolasdmolina)! - Fix first-run install failing with "ACN candidate … no longer available" when daemon startup phases (Resolving / PreparingBackend / Starting) hold a stable progress key longer than 30s

- [`ba17e6f`](https://github.com/magnitudedev/magnitude/commit/ba17e6f967fc9ed3eaa991d8a8da547cd6867e3f) Thanks [@anerli](https://github.com/anerli)! - fix tui defects

- [`fedb8dc`](https://github.com/magnitudedev/magnitude/commit/fedb8dc8b56315d8b799d4bd4f556ee16b6db935) Thanks [@anerli](https://github.com/anerli)! - fix: remove unnecessary acn coordination historical revision check

- [`ba17e6f`](https://github.com/magnitudedev/magnitude/commit/ba17e6f967fc9ed3eaa991d8a8da547cd6867e3f) Thanks [@anerli](https://github.com/anerli)! - words

- [`ba17e6f`](https://github.com/magnitudedev/magnitude/commit/ba17e6f967fc9ed3eaa991d8a8da547cd6867e3f) Thanks [@anerli](https://github.com/anerli)! - response format overhaul, grammar, provider, etc

- [`ba17e6f`](https://github.com/magnitudedev/magnitude/commit/ba17e6f967fc9ed3eaa991d8a8da547cd6867e3f) Thanks [@anerli](https://github.com/anerli)! - advisor improvements

- [`ba17e6f`](https://github.com/magnitudedev/magnitude/commit/ba17e6f967fc9ed3eaa991d8a8da547cd6867e3f) Thanks [@anerli](https://github.com/anerli)! - provider

- [`ba17e6f`](https://github.com/magnitudedev/magnitude/commit/ba17e6f967fc9ed3eaa991d8a8da547cd6867e3f) Thanks [@anerli](https://github.com/anerli)! - pass NO_COLOR=1 for shell tool

- [`7a49f4a`](https://github.com/magnitudedev/magnitude/commit/7a49f4ae329713d9955aed3951eb6d1caf9cfdf2) Thanks [@anerli](https://github.com/anerli)! - download/load endpoint termination fixes

- [`ba17e6f`](https://github.com/magnitudedev/magnitude/commit/ba17e6f967fc9ed3eaa991d8a8da547cd6867e3f) Thanks [@anerli](https://github.com/anerli)! - advisor

- [`ba17e6f`](https://github.com/magnitudedev/magnitude/commit/ba17e6f967fc9ed3eaa991d8a8da547cd6867e3f) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - file picker

- [`d3b10cb`](https://github.com/magnitudedev/magnitude/commit/d3b10cb558907fbce6de26a60b257574d0a473fe) Thanks [@anerli](https://github.com/anerli)! - fix cuda resolution issues

- [`847d612`](https://github.com/magnitudedev/magnitude/commit/847d612bf97fd27e5598d0a3d93f06d50957fa1c) Thanks [@anerli](https://github.com/anerli)! - local

- [`c7d2298`](https://github.com/magnitudedev/magnitude/commit/c7d2298b503f7d8852572488746a6adbca43120f) Thanks [@anerli](https://github.com/anerli)! - improve icn lifecycle error specificity

- [`72dbf86`](https://github.com/magnitudedev/magnitude/commit/72dbf8693d76d0c2083704a2a4dcc187b1920b3a) Thanks [@anerli](https://github.com/anerli)! - daemon fixes

- [`262f85d`](https://github.com/magnitudedev/magnitude/commit/262f85d7a4b220732819ac861603bf862e1c1d43) Thanks [@anerli](https://github.com/anerli)! - fix inference workers missing native libs

- [`6401c9c`](https://github.com/magnitudedev/magnitude/commit/6401c9c0346a98e47cd8a3de31452bd6ac36ff3d) Thanks [@anerli](https://github.com/anerli)! - fix linux arm64 dynamic bindings

- [`ba17e6f`](https://github.com/magnitudedev/magnitude/commit/ba17e6f967fc9ed3eaa991d8a8da547cd6867e3f) Thanks [@anerli](https://github.com/anerli)! - fix: turn off baml logs env var on cli entry

- [`9151adc`](https://github.com/magnitudedev/magnitude/commit/9151adca5170626bebd76c213a3f7196c44cc174) Thanks [@nicolasdmolina](https://github.com/nicolasdmolina)! - fix: do not retire live Starting ACN on stable progress key

- [`ba17e6f`](https://github.com/magnitudedev/magnitude/commit/ba17e6f967fc9ed3eaa991d8a8da547cd6867e3f) Thanks [@anerli](https://github.com/anerli)! - Switch Linux x64 binary to bun-linux-x64-baseline target to fix SIGILL crash on CPUs without BMI2 support

- [`90ffcb5`](https://github.com/magnitudedev/magnitude/commit/90ffcb5787ab67249b1b3e0741a3bf9c44fdd2e3) Thanks [@anerli](https://github.com/anerli)! - local

- [`ba17e6f`](https://github.com/magnitudedev/magnitude/commit/ba17e6f967fc9ed3eaa991d8a8da547cd6867e3f) Thanks [@anerli](https://github.com/anerli)! - various improvements

- [`ba17e6f`](https://github.com/magnitudedev/magnitude/commit/ba17e6f967fc9ed3eaa991d8a8da547cd6867e3f) Thanks [@anerli](https://github.com/anerli)! - fixes

- [`ba17e6f`](https://github.com/magnitudedev/magnitude/commit/ba17e6f967fc9ed3eaa991d8a8da547cd6867e3f) Thanks [@anerli](https://github.com/anerli)! - task system

- [`ba17e6f`](https://github.com/magnitudedev/magnitude/commit/ba17e6f967fc9ed3eaa991d8a8da547cd6867e3f) Thanks [@anerli](https://github.com/anerli)! - fix: auth refresh coordination with multiple sessions

- [`b0e72cc`](https://github.com/magnitudedev/magnitude/commit/b0e72cc96232d81967eeb969468a51de30ca643a) Thanks [@anerli](https://github.com/anerli)! - fix: build linux with ubuntu 22 to lower floor of required glibc support

- [`497c151`](https://github.com/magnitudedev/magnitude/commit/497c151272dc7ae5bb6892ed746b74a3bed3900b) Thanks [@anerli](https://github.com/anerli)! - fix: remove native openssl implicit dep

- [`311dcb6`](https://github.com/magnitudedev/magnitude/commit/311dcb68e1186b8f4df039c95717577ad7cbdbfb) Thanks [@anerli](https://github.com/anerli)! - fix inference worker timeout

- [`7ea43d0`](https://github.com/magnitudedev/magnitude/commit/7ea43d07185933a182ae95d1680a675c6b206089) Thanks [@anerli](https://github.com/anerli)! - allow launcher to use node or bun

- [`ba17e6f`](https://github.com/magnitudedev/magnitude/commit/ba17e6f967fc9ed3eaa991d8a8da547cd6867e3f) Thanks [@anerli](https://github.com/anerli)! - fix cli entry

- [`ba17e6f`](https://github.com/magnitudedev/magnitude/commit/ba17e6f967fc9ed3eaa991d8a8da547cd6867e3f) Thanks [@anerli](https://github.com/anerli)! - fixes

- [`ba17e6f`](https://github.com/magnitudedev/magnitude/commit/ba17e6f967fc9ed3eaa991d8a8da547cd6867e3f) Thanks [@anerli](https://github.com/anerli)! - compaction fixes

- [`ba17e6f`](https://github.com/magnitudedev/magnitude/commit/ba17e6f967fc9ed3eaa991d8a8da547cd6867e3f) Thanks [@anerli](https://github.com/anerli)! - fixes

- [`ba17e6f`](https://github.com/magnitudedev/magnitude/commit/ba17e6f967fc9ed3eaa991d8a8da547cd6867e3f) Thanks [@anerli](https://github.com/anerli)! - better shell tool

- [`f76256c`](https://github.com/magnitudedev/magnitude/commit/f76256c00739bf172e6d6a91da922345cba52e08) Thanks [@anerli](https://github.com/anerli)! - fix timeouts

- [`94ecdde`](https://github.com/magnitudedev/magnitude/commit/94ecdde39a164c5791950a5b92fd106261a3493d) Thanks [@anerli](https://github.com/anerli)! - installation and acn lifecycle fixes

- [`ba17e6f`](https://github.com/magnitudedev/magnitude/commit/ba17e6f967fc9ed3eaa991d8a8da547cd6867e3f) Thanks [@anerli](https://github.com/anerli)! - init

- [`ba17e6f`](https://github.com/magnitudedev/magnitude/commit/ba17e6f967fc9ed3eaa991d8a8da547cd6867e3f) Thanks [@anerli](https://github.com/anerli)! - vcs fixes, load skills

- [`ba17e6f`](https://github.com/magnitudedev/magnitude/commit/ba17e6f967fc9ed3eaa991d8a8da547cd6867e3f) Thanks [@anerli](https://github.com/anerli)! - fix ripgrep packaging

- [`ba17e6f`](https://github.com/magnitudedev/magnitude/commit/ba17e6f967fc9ed3eaa991d8a8da547cd6867e3f) Thanks [@anerli](https://github.com/anerli)! - fix: lead hang on subagent idle with no message

- [`ba17e6f`](https://github.com/magnitudedev/magnitude/commit/ba17e6f967fc9ed3eaa991d8a8da547cd6867e3f) Thanks [@anerli](https://github.com/anerli)! - fix render crash

- [`ba17e6f`](https://github.com/magnitudedev/magnitude/commit/ba17e6f967fc9ed3eaa991d8a8da547cd6867e3f) Thanks [@anerli](https://github.com/anerli)! - fix distributed executable name

- [`76f39df`](https://github.com/magnitudedev/magnitude/commit/76f39df010704575e8881e04e21d7fe8d45c4360) Thanks [@anerli](https://github.com/anerli)! - fix: preserve logic shard paths

- [`ba17e6f`](https://github.com/magnitudedev/magnitude/commit/ba17e6f967fc9ed3eaa991d8a8da547cd6867e3f) Thanks [@anerli](https://github.com/anerli)! - Parse file mentions in CLI options with support for line ranges and support line ranges in the TUI file picker.

- [`041bd76`](https://github.com/magnitudedev/magnitude/commit/041bd764075fbad1965afebeb7881a4046e74003) Thanks [@anerli](https://github.com/anerli)! - fix: ensure hardware calibration during ICN startup and other calibration issues

- [`ba17e6f`](https://github.com/magnitudedev/magnitude/commit/ba17e6f967fc9ed3eaa991d8a8da547cd6867e3f) Thanks [@anerli](https://github.com/anerli)! - update package info, use root README

- [`cd89dfc`](https://github.com/magnitudedev/magnitude/commit/cd89dfcca87c5862374b4daa910fb2124f0975fb) Thanks [@anerli](https://github.com/anerli)! - timeout fixes, daemon coordination fixes

- [`e948b25`](https://github.com/magnitudedev/magnitude/commit/e948b256416245846b7c4f60695b0c97076f650d) Thanks [@anerli](https://github.com/anerli)! - fix: support macOS 13 and above instead of only 15 and above

## 0.0.1-alpha.41

### Patch Changes

- [`e948b25`](https://github.com/magnitudedev/magnitude/commit/e948b256416245846b7c4f60695b0c97076f650d) Thanks [@anerli](https://github.com/anerli)! - fix: support macOS 13 and above instead of only 15 and above

## 0.0.1-alpha.40

### Patch Changes

- [`fedb8dc`](https://github.com/magnitudedev/magnitude/commit/fedb8dc8b56315d8b799d4bd4f556ee16b6db935) Thanks [@anerli](https://github.com/anerli)! - fix: remove unnecessary acn coordination historical revision check

- [`c7d2298`](https://github.com/magnitudedev/magnitude/commit/c7d2298b503f7d8852572488746a6adbca43120f) Thanks [@anerli](https://github.com/anerli)! - improve icn lifecycle error specificity

## 0.0.1-alpha.39

### Patch Changes

- [#19](https://github.com/magnitudedev/magnitude/pull/19) [`fcdd491`](https://github.com/magnitudedev/magnitude/commit/fcdd491ad6e813feb47601b084b4187645be123a) Thanks [@nicolasdmolina](https://github.com/nicolasdmolina)! - Fix first-run install failing with "ACN candidate … no longer available" when daemon startup phases (Resolving / PreparingBackend / Starting) hold a stable progress key longer than 30s

- [`9151adc`](https://github.com/magnitudedev/magnitude/commit/9151adca5170626bebd76c213a3f7196c44cc174) Thanks [@nicolasdmolina](https://github.com/nicolasdmolina)! - fix: do not retire live Starting ACN on stable progress key

## 0.0.1-alpha.38

### Patch Changes

- [`b0e72cc`](https://github.com/magnitudedev/magnitude/commit/b0e72cc96232d81967eeb969468a51de30ca643a) Thanks [@anerli](https://github.com/anerli)! - fix: build linux with ubuntu 22 to lower floor of required glibc support

## 0.0.1-alpha.37

### Patch Changes

- [`311dcb6`](https://github.com/magnitudedev/magnitude/commit/311dcb68e1186b8f4df039c95717577ad7cbdbfb) Thanks [@anerli](https://github.com/anerli)! - fix inference worker timeout

## 0.0.1-alpha.36

### Patch Changes

- [`847d612`](https://github.com/magnitudedev/magnitude/commit/847d612bf97fd27e5598d0a3d93f06d50957fa1c) Thanks [@anerli](https://github.com/anerli)! - local

## 0.0.1-alpha.35

### Patch Changes

- [`cd89dfc`](https://github.com/magnitudedev/magnitude/commit/cd89dfcca87c5862374b4daa910fb2124f0975fb) Thanks [@anerli](https://github.com/anerli)! - timeout fixes, daemon coordination fixes

## 0.0.1-alpha.34

### Patch Changes

- [`f76256c`](https://github.com/magnitudedev/magnitude/commit/f76256c00739bf172e6d6a91da922345cba52e08) Thanks [@anerli](https://github.com/anerli)! - fix timeouts

## 0.0.1-alpha.33

### Patch Changes

- [`76f39df`](https://github.com/magnitudedev/magnitude/commit/76f39df010704575e8881e04e21d7fe8d45c4360) Thanks [@anerli](https://github.com/anerli)! - fix: preserve logic shard paths

## 0.0.1-alpha.32

### Patch Changes

- [`041bd76`](https://github.com/magnitudedev/magnitude/commit/041bd764075fbad1965afebeb7881a4046e74003) Thanks [@anerli](https://github.com/anerli)! - fix: ensure hardware calibration during ICN startup and other calibration issues

## 0.0.1-alpha.31

### Patch Changes

- [`7a49f4a`](https://github.com/magnitudedev/magnitude/commit/7a49f4ae329713d9955aed3951eb6d1caf9cfdf2) Thanks [@anerli](https://github.com/anerli)! - download/load endpoint termination fixes

## 0.0.1-alpha.30

### Patch Changes

- [`72dbf86`](https://github.com/magnitudedev/magnitude/commit/72dbf8693d76d0c2083704a2a4dcc187b1920b3a) Thanks [@anerli](https://github.com/anerli)! - daemon fixes

## 0.0.1-alpha.29

### Patch Changes

- [`7ea43d0`](https://github.com/magnitudedev/magnitude/commit/7ea43d07185933a182ae95d1680a675c6b206089) Thanks [@anerli](https://github.com/anerli)! - allow launcher to use node or bun

## 0.0.1-alpha.28

### Patch Changes

- [`d3b10cb`](https://github.com/magnitudedev/magnitude/commit/d3b10cb558907fbce6de26a60b257574d0a473fe) Thanks [@anerli](https://github.com/anerli)! - fix cuda resolution issues

## 0.0.1-alpha.27

### Patch Changes

- [`497c151`](https://github.com/magnitudedev/magnitude/commit/497c151272dc7ae5bb6892ed746b74a3bed3900b) Thanks [@anerli](https://github.com/anerli)! - fix: remove native openssl implicit dep

## 0.0.1-alpha.26

### Patch Changes

- [`262f85d`](https://github.com/magnitudedev/magnitude/commit/262f85d7a4b220732819ac861603bf862e1c1d43) Thanks [@anerli](https://github.com/anerli)! - fix inference workers missing native libs

## 0.0.1-alpha.25

### Patch Changes

- [`94ecdde`](https://github.com/magnitudedev/magnitude/commit/94ecdde39a164c5791950a5b92fd106261a3493d) Thanks [@anerli](https://github.com/anerli)! - installation and acn lifecycle fixes

## 0.0.1-alpha.24

### Patch Changes

- [`6401c9c`](https://github.com/magnitudedev/magnitude/commit/6401c9c0346a98e47cd8a3de31452bd6ac36ff3d) Thanks [@anerli](https://github.com/anerli)! - fix linux arm64 dynamic bindings

## 0.0.1-alpha.23

### Patch Changes

- [`90ffcb5`](https://github.com/magnitudedev/magnitude/commit/90ffcb5787ab67249b1b3e0741a3bf9c44fdd2e3) Thanks [@anerli](https://github.com/anerli)! - local

## 0.0.1-alpha.22

### Patch Changes

- [`a768913`](https://github.com/magnitudedev/agent/commit/a76891391d06d2b0ed2a298ce5dbe36b1e0104da) Thanks [@anerli](https://github.com/anerli)! - Switch Linux x64 binary to bun-linux-x64-baseline target to fix SIGILL crash on CPUs without BMI2 support

## 0.0.1-alpha.21

### Patch Changes

- [`4be25fe`](https://github.com/magnitudedev/agent/commit/4be25fe96942a56d2895923fc3f9d5b872a269d8) Thanks [@anerli](https://github.com/anerli)! - fix tui defects

## 0.0.1-alpha.20

### Patch Changes

- [`29bfbe4`](https://github.com/magnitudedev/agent/commit/29bfbe478e2899cfeb5c6d592a0ab4e05a8b237a) Thanks [@anerli](https://github.com/anerli)! - words

## 0.0.1-alpha.19

### Patch Changes

- [`77909b9`](https://github.com/magnitudedev/agent/commit/77909b95ccd091aa7e970e5c6b9773aa83d982ae) Thanks [@anerli](https://github.com/anerli)! - rm advisor

## 0.0.1-alpha.18

### Patch Changes

- [`4859b4d`](https://github.com/magnitudedev/agent/commit/4859b4d4d4ba326e8fca593ede00354b5532a6b5) Thanks [@anerli](https://github.com/anerli)! - advisor improvements

## 0.0.1-alpha.17

### Patch Changes

- [`81a5f3b`](https://github.com/magnitudedev/agent/commit/81a5f3bdfea1d70069bc49722d8b16d9a28de3cf) Thanks [@anerli](https://github.com/anerli)! - fix ripgrep packaging

## 0.0.1-alpha.16

### Patch Changes

- [`a624e85`](https://github.com/magnitudedev/agent/commit/a624e85c0d7abd17d656f4703b75de596d6ae3b0) Thanks [@anerli](https://github.com/anerli)! - vcs fixes, load skills

## 0.0.1-alpha.15

### Patch Changes

- [`9875960`](https://github.com/magnitudedev/magnitude/commit/98759603110c688dfcc4342af350645bb50270c2) Thanks [@anerli](https://github.com/anerli)! - advisor

- [#25](https://github.com/magnitudedev/magnitude/pull/25) [`653c925`](https://github.com/magnitudedev/magnitude/commit/653c9259bb9d46368ac1edb6ae5fae77fe4580cb) Thanks [@ewired](https://github.com/ewired)! - Parse file mentions in CLI options with support for line ranges and support line ranges in the TUI file picker.

## 0.0.1-alpha.14

### Patch Changes

- [`8c51027`](https://github.com/magnitudedev/magnitude/commit/8c51027c7af629ca91f9a5c507c273729e3847e3) Thanks [@anerli](https://github.com/anerli)! - fix render crash

## 0.0.1-alpha.13

### Patch Changes

- [`79eeafc`](https://github.com/magnitudedev/magnitude/commit/79eeafcbb73970e50d4713c930a4fc78cb2980c7) Thanks [@anerli](https://github.com/anerli)! - response format overhaul, grammar, provider, etc

- [`fa3975e`](https://github.com/magnitudedev/magnitude/commit/fa3975e80ef972c49b67daef3382b1f6588cba74) Thanks [@anerli](https://github.com/anerli)! - provider

## 0.0.1-alpha.12

### Patch Changes

- [`211166d`](https://github.com/magnitudedev/magnitude/commit/211166db38500080d669e0dba53ab87d286c0737) Thanks [@anerli](https://github.com/anerli)! - fixes

## 0.0.1-alpha.11

### Patch Changes

- [`ce95b6b`](https://github.com/magnitudedev/magnitude/commit/ce95b6bddc06b346a4af5fff732525c770af94f8) Thanks [@anerli](https://github.com/anerli)! - fixes

## 0.0.1-alpha.10

### Patch Changes

- [`fe9a864`](https://github.com/magnitudedev/magnitude/commit/fe9a864c8fe669b3f875f25478a4dd6964f7641e) Thanks [@anerli](https://github.com/anerli)! - fixes

## 0.0.1-alpha.9

### Patch Changes

- [`ceaa3d3`](https://github.com/magnitudedev/magnitude/commit/ceaa3d38983061481c89a1e31c4bb402b3c3bfae) Thanks [@anerli](https://github.com/anerli)! - task system

## 0.0.1-alpha.8

### Patch Changes

- [`283de77`](https://github.com/magnitudedev/magnitude/commit/283de77fbf444a4fb3608bc915cb6209b9694c39) Thanks [@anerli](https://github.com/anerli)! - compaction fixes

## 0.0.1-alpha.7

### Patch Changes

- [`94f637d`](https://github.com/magnitudedev/magnitude/commit/94f637ded5c390cca0490c43d8fe829ab9e44ae9) Thanks [@anerli](https://github.com/anerli)! - fix: lead hang on subagent idle with no message

## 0.0.1-alpha.6

### Patch Changes

- [`4ba9d87`](https://github.com/magnitudedev/magnitude/commit/4ba9d877ecf1264dab209a6263c9c9885b79f780) Thanks [@anerli](https://github.com/anerli)! - various improvements

## 0.0.1-alpha.5

### Patch Changes

- [`5620e60`](https://github.com/magnitudedev/magnitude/commit/5620e60904710fad26937366460d7c2bcb1dad79) Thanks [@anerli](https://github.com/anerli)! - better shell tool

## 0.0.1-alpha.4

### Patch Changes

- [`cdfd178`](https://github.com/magnitudedev/magnitude/commit/cdfd178cb108314d75ce57ea7bc3ce779719fe82) Thanks [@anerli](https://github.com/anerli)! - pass NO_COLOR=1 for shell tool

- [`b34cba1`](https://github.com/magnitudedev/magnitude/commit/b34cba17b094a330b7a4ec30a8c51f7ce3fd5eba) Thanks [@thrgreenwald](https://github.com/thrgreenwald)! - file picker

- [`241a629`](https://github.com/magnitudedev/magnitude/commit/241a6293d33dbe2b8edc3446ed40e65cff64e8be) Thanks [@anerli](https://github.com/anerli)! - fix: auth refresh coordination with multiple sessions

## 0.0.1-alpha.3

### Patch Changes

- [`0cead3d`](https://github.com/magnitudedev/magnitude/commit/0cead3deff16482e654d5f38192a84e78da95861) Thanks [@anerli](https://github.com/anerli)! - fix: turn off baml logs env var on cli entry

- [`948ecb1`](https://github.com/magnitudedev/magnitude/commit/948ecb10f22424f057589675481049cf39fa7576) Thanks [@anerli](https://github.com/anerli)! - update package info, use root README

## 0.0.1-alpha.2

### Patch Changes

- [`0093019`](https://github.com/magnitudedev/magnitude/commit/00930198daf8759ee705b50587ec7ce4d2a1d1f1) Thanks [@anerli](https://github.com/anerli)! - fix distributed executable name

## 0.0.1-alpha.1

### Patch Changes

- [`2cb6b70`](https://github.com/magnitudedev/magnitude/commit/2cb6b705126dac832bffad3e98dbfe3079fa4e54) Thanks [@anerli](https://github.com/anerli)! - fix cli entry

## 0.0.1-alpha.0

### Patch Changes

- [`3f9b563`](https://github.com/magnitudedev/magnitude/commit/3f9b563cb408ccd8deddcf23c773cce8cb589763) Thanks [@anerli](https://github.com/anerli)! - init
