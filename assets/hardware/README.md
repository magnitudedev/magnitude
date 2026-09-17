# Hardware presentation catalog

The [complete mapping](MAPPING.md) lists every bundled photo, shared exterior group, exact firmware/GPU identifier, and processor constraint. `inventory.json` is the source of truth for artwork; `facts.json` holds a small catalog of published chip specifications. Provenance stays in these maintenance files, with no source links or credit UI in the hardware card.

## Identification and meaningful details

Identity, physical enclosure, and inference capability are separate facts. A laptop can use CPU inference, integrated graphics, a discrete mobile GPU, or an external GPU. A shared-memory chip such as Ryzen AI Max does not identify whether it is installed in a tablet, mobile workstation, or mini PC. NVIDIA GB10 likewise does not establish a DGX Spark enclosure: other manufacturers sell GB10 systems.

The privileged desktop host reads only public manufacturer/product/family/version and chassis type. Windows uses bounded SMBIOS Type 1 and Type 3 records; Linux reads the corresponding public DMI files; macOS reads `hw.model`. No serial number, UUID, asset tag, subprocess hardware inventory, benchmark, or network lookup is added. Firmware chassis type still supplies a generic category when product names are unavailable or unrecognized.

A verified enclosure match wins over a graphics-card image. Unknown portable and all-in-one machines never inherit a desktop-card photo. Desktop GPU fallback requires an observed dedicated physical memory domain and an exact GPU name; mobile, Ti, SUPER, XT/XTX and regional suffixes remain significant. Card artwork represents a GPU family, not detected board-partner or cooler identity. Shared exterior designs use one file, including the identical PowerColor reference artwork for RX 6800 XT/6900 XT and Hellhound artwork for RX 7900 XT/XTX.

Known PC matches include Dell XPS, Inspiron and G15; HP Pavilion, Envy and EliteBook; Lenovo ThinkPad; ASUS ROG Zephyrus and Flow; NVIDIA DGX Spark; Beelink GTR; Minisforum MS-S1 MAX; HP Z2 Mini and ZBook Ultra. These are exact supported product identities, not wildcards for all products from those vendors. XPS 15 9530 and Inspiron 3520 require a matching modern CPU token because Dell reused those model names. Lenovo exposes its readable product name in firmware version, separate from its machine-type product code.

The card displays observed system or unified memory, a single CPU core count, and each accelerator's shared-memory relationship or dedicated VRAM. Shared memory is not added to system RAM; duplicate API views of one physical memory domain do not duplicate its VRAM total. Distinct GPUs keep separate capacities. “CPU inference” means no accelerator was reported by the inference service, not that the computer physically lacks graphics hardware.

The CPU count prefers observed physical cores and falls back to an exact fixed catalog count; ambiguous options and server per-processor totals are omitted. Threads and duplicate CPU specification counts are not displayed. Other published facts are matched to exact identities. Visible labels omit the internal `(spec)` suffix, with no tooltip. They are static nominal specifications, not benchmark results or free/available memory. The catalog deliberately omits uncertain bin-dependent GPU counts and bandwidth, including ambiguous Apple Max variants. Physical CPU topology can resolve only explicitly cataloged unique bins. Unknown chips still show observed details. Catalog lookup adds no native discovery or runtime networking. The optional physical-core query is cached once per ICN process; [DETECTION.md](DETECTION.md) documents implementations, measured local query costs, and researched GPU APIs.

[FACTS.md](FACTS.md) lists every published specification and all 44 researched photo-group mappings. [coverage.json](coverage.json) records factory CPU options and configuration limits; tests fail for missing groups, missing processor core information, missing named GPU facts, or ambiguous catalog aliases. Regenerate the human-readable tables with `python3 assets/hardware/tools/spec_audit.py`.

## Photography and validation

[Photo size audit](OPTIMIZATION.md) covers all 44 assets: 21.03 MB reduced to 7.57 MB using the same photography, lossless WebP encoding and selective resizing for the 260-pixel card. Transparency and ICC profiles are preserved.

Use a single complete, clean product silhouette with real transparency, enough resolution for the card on high-density displays, and no boxes, award badges, annotations, background scene, or front/back composite. Never infer a physical enclosure from a chip name to increase photo coverage. Each inventory entry records the nontransparent pixel bounds. The shared photo renderer fits those bounds into a 4:3 frame with 6% inset padding, ignoring transparent file margins while preserving the complete silhouette and shadows. The gallery uses the same renderer. Update framing whenever an asset changes; the photo audit rejects stale bounds. Native manufacturer cutouts are preferred; keep original source URLs and identity evidence in the inventory.

Run `python assets/hardware/tools/audit.py` with Pillow to check image decoding, dimensions, real transparency, file coverage, and duplicate bytes, and regenerate the mapping and audit records. Visually inspect new images on both light and dark surfaces; an alpha channel alone does not prove a clean silhouette.

From `desktop`, run `bunx --bun vitest run src/hardware-photos.test.ts src/hardware-details.test.ts`. The real hardware component has acceptance fixtures at `desktop/test/hardware/index.html` (append `?dark` for dark mode), served by the existing loading-test Vite server. Test every supported identity plus unknown, shared-memory, laptop-GPU, external-GPU, multiple-GPU, and ambiguous-name cases.

Native parser fixtures can be run from the repository root:

```sh
cc -std=c11 -Wall -Wextra -Werror -fsanitize=address,undefined packages/daemon-management/native/test/machine-identity.c -o /tmp/magnitude-machine-identity-test
/tmp/magnitude-machine-identity-test
bunx --bun vitest run packages/daemon-management/src/desktop-native/machine-identity.test.ts
```
