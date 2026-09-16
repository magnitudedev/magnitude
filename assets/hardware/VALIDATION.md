# Hardware catalog verification

Verified 2026-09-15/16 during the hardware presentation expansion.

## Coverage delivered

44 distinct photos: 28 enclosure groups and 16 graphics-card groups. The expansion adds 14 exact PC enclosure groups across consumer/business laptops, gaming laptops/tablets, mobile workstations and compact AI PCs. Two existing pairs of GPU entries now share the manufacturer's identical exterior image instead of bundling duplicate bytes. [MAPPING.md](MAPPING.md) lists all matching identifiers and constraints; [inventory.json](inventory.json) retains source evidence.

Every photo was decoded and checked for dimensions, transparency and unique bytes. All 44 have real alpha transparency. Every image was also inspected composited on white and dark backgrounds. That visual review rejected two old Apple renders whose alpha channels surrounded opaque white rectangles; clean Apple cutouts replaced them. Nine old JPEG assets were replaced by transparent product renders. Generated background-removal attempts that returned RGB checkerboards were rejected and are not bundled. No generated product artwork is included.

Manufacturer images are preferred. Retired Surface renders come from Grover's product listings. The 2013 Mac Pro render comes from an Apple press-image archive preserved by The Apple Wiki; the inventory retains that provenance. Product artwork is representative of an exterior/card family and does not claim to detect color or board partner.

Research also considered ASUS Ascent GX10, Framework Desktop and GMKtec EVO-X2. Their reviewed candidate images or available identity evidence did not meet both the photo-quality and exact-enclosure matching criteria for this change. They continue to receive observed hardware details and a firmware category, without a guessed enclosure photograph. Strix Halo and GB10 are never treated as unique enclosure identifiers. This is a curated catalog, not a claim to recognize every computer on the market.

## Checks

- Photo audit: 44 distinct, decodable, transparent assets, each at least 500 pixels on its longest dimension; hashes and exact dimensions in [photo-audit.json](photo-audit.json). Actual foreground size and edges were inspected visually as well.
- Focused desktop tests: 91 passed, covering every inventory match, unknown/misleading identities, mobile suffixes, XPS/Inspiron model reuse, driver-name decorations, observed physical cores versus available threads, Apple configuration selection, Lenovo version identity, shared memory, multiple GPUs, duplicate backend memory domains and exact published-spec matches.
- Native CPU topology/hardware tests: 49 passed; ACN observation projection tests: 4 passed. Generated ICN protocol consistency check and targeted ACN protocol TypeScript check passed.
- Native identity tests: 28 passed, including a live macOS native binding observation.
- SMBIOS C fixtures: passed under AddressSanitizer and UndefinedBehaviorSanitizer, including reversed record order, every truncated prefix, out-of-range string indices, oversized strings, chassis lock bits, terminal records and 10,000 deterministic malformed inputs.
- Production desktop build: passed, including native addon compilation and renderer asset bundling.
- Targeted desktop TypeScript check: no TypeScript errors or diagnostics in the changed files. The patched compiler exits 2 because of existing Effect warnings/messages elsewhere (including `main.ts`, SDK client and storage); this is not recorded as a clean zero-exit type-check.
- Full desktop suite: 180 passed, 5 skipped, one unrelated `shell-env.test.ts` process-timing failure (`ENOENT` for its temporary PID file). Its complete seven-test file passed in isolation. An earlier concurrent run had the same class of shell timing failure. Hardware tests passed in both runs.
- Browser acceptance: all eight actual hardware cards load their images without horizontal card overflow. Inspected light mode at 800 and 1600 pixels and dark mode at 800 and 1120 pixels, including the two-GPU and shared-memory layouts. Temporary browser viewport was restored.
- `git diff --check` and design-document applicability check passed.

Actual Windows/Linux machines were not available for an end-to-end native observation; Windows SMBIOS behavior is covered by bounded parser fixtures and Linux public-DMI reading is reviewed code. No benchmark, latency claim, or live verification on those operating systems is implied.

## Published facts

The expanded catalog contains 113 entries: 74 processors, 38 accelerators and one device specification. Every processor has either a fixed core count (65) or explicit configurable counts (9). GPU unit counts are populated for 36 of 38 accelerator entries; RTX 3050 Laptop GPU variants and the anomalous Apple FirePro D500 count remain explicitly unresolved. Every one of the 44 photo groups has a researched coverage record, including its CPU/GPU configurations and matching limits. [FACTS.md](FACTS.md) contains the complete generated mapping and primary-source links.

Physical CPU core detection is now implemented and cached once per ICN process. Installed memory and available CPU threads remain observations, separate from nominal published values. A local Mac sysctl timing experiment measured a median of 0.916 microseconds. GPU core-query research includes a successful local Apple prototype, but that supplemental GPU query is not integrated; NVIDIA/AMD candidates still require platform validation. [DETECTION.md](DETECTION.md) records the implementation, experiments and limitations. No runtime catalog network requests or benchmarks were added.

The expanded eight-card fixture was inspected again after the facts and CPU observation changes. The earlier full-suite and native identity checks above predate the final catalog expansion; the 91 focused desktop tests, 49 Rust hardware tests, four projection tests, generated protocol check and production build were rerun for the expanded implementation.

## GPU bandwidth follow-up (2026-09-16)

Bandwidth now covers all 34 discrete accelerator entries (including all 22 NVIDIA entries) in the 115-entry catalog. Six shared-memory Radeon entries have no independent fixed VRAM bandwidth. Four GPU families select capacity-dependent values from observed dedicated VRAM: RTX 3060, RTX 3050 Laptop, RTX A2000 Laptop and Arc A770. A100 PCIe/SXM and 40/80 GB names select their distinct published values. Unknown capacities omit conditional facts. Mobile values describe nominal maxima; OEM settings and power states can reduce bandwidth.

Sources are retained with every entry. Most are manufacturer specifications; mobile 4050/3050 values also use published experimental hardware tables. The RTX 4060 Laptop value is supported by PurePC’s original ASUS TUF hardware test; the RTX 4080 Laptop value is derived from Notebookcheck’s original Legion Pro 7 test configuration: 2250 MHz GDDR6 (18 Gbit/s effective) × 192 bits ÷ 8 = 432 GB/s. These describe hardware specifications, not measured bandwidth results.

CUDA cores, compute units, Xe cores and stream processors remain in the reviewable research catalog but are omitted from user hardware cards. No tooltip was added. Browser inspection confirms the gaming-laptop card displays bandwidth, the Strix Halo card omits compute units, and the gallery has no horizontal overflow or visible `(spec)` suffix.

Follow-up validation: 114 focused hardware tests passed. Targeted desktop type checking reported no TypeScript errors or diagnostics in the changed files; existing Effect diagnostics elsewhere still produce exit status 2. `git diff --check` passed.

## Photo size optimization (2026-09-16)

All 44 existing photos were audited and visually compared before/after on light and dark surfaces. Selective 960-pixel resizing and lossless WebP reduced bundled bytes from 21,025,654 to 7,572,930 (64.0%). Eighteen images were resized; 26 retain identical decoded pixels, including four left unchanged. Alpha and embedded ICC profiles were verified. All 114 hardware tests, the photo audit and production Electron/Vite build passed. All 44 built photo hashes match their optimized sources; every gallery photo loaded successfully. [OPTIMIZATION.md](OPTIMIZATION.md) lists every decision and measured size.
