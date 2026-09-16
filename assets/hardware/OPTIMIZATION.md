# Hardware photo size audit

Audited all 44 existing photos on 2026-09-16. Total: **21.03 MB → 7.57 MB**, saving **13.45 MB (64.0%)**. MB/KB are decimal. These are bundled asset bytes, not an installer measurement.

No replacement photography. Compare the original encoding, lossless WebP at full resolution, and a lossless-WebP version with a 960-pixel longest edge. Keep the smallest acceptable version. The production card is capped at 260 CSS pixels; the 960-pixel cap retains more than 3× resolution. Existing smaller images are never enlarged. Preserve transparency and embedded ICC profiles.

Eighteen images were resized; the other 26 retain identical decoded RGBA pixels, including four unchanged files. Resizing removes source detail, so the complete optimization is not pixel-lossless; WebP encoding itself is lossless. All 44 original/optimized pairs were inspected on white and dark backgrounds at card scale. At 520×390 (2× the largest card), resized selections have at least 52.84 dB PSNR and at most 0.582/255 channel RMSE against the original rendered at the same size. These metrics support, but do not replace, visual review. Dell G15 retains its original dimensions after the resized candidate changed raster alignment.

Lossless conversion alone would reduce the collection to about 14 MB. Existing compact WebP files are kept when re-encoding or resizing increases their size. The largest remaining asset is the detailed Dell Inspiron display photograph; more aggressive savings would require lossy encoding or a lower resolution.

## Subsequent quality upgrade

The 2024 Mac mini was subsequently replaced with Apple’s original October 2024 high-resolution render. Its visible hardware is now 965 × 387 pixels instead of 280 × 112. The transparent lossless WebP is 161,464 bytes; the current collection totals 7,721,706 bytes (7.72 MB). The table below and `optimization-audit.json` describe the earlier compression pass; `photo-audit.json` records the current assets.

Mac Studio was also upgraded to Apple’s original high-resolution front render, retaining 964 × 473 pixels of visible hardware (previously 434 × 212) in a 374,802-byte lossless WebP. After both upgrades, all 44 photos total 8,060,572 bytes (8.06 MB).

## Every photo

| Photo | Original dimensions | Selected dimensions | Before KB | After KB | Saved | Action |
| --- | --- | --- | ---: | ---: | ---: | --- |
| [Dell Inspiron 15 3520](dell-inspiron-open.webp) | 1600×1062 | 960×637 | 2574.2 | 732.1 | 71.6% | Resize + lossless WebP |
| [Arc A770](arc-a770.webp) | 2000×2000 | 960×960 | 2040.9 | 310.5 | 84.8% | Resize + lossless WebP |
| [Arc B580](arc-b580.webp) | 2000×2000 | 960×960 | 1810.0 | 309.7 | 82.9% | Resize + lossless WebP |
| [Mac Pro](mac-pro-2013.webp) | 1260×2195 | 551×960 | 1463.5 | 265.8 | 81.8% | Resize + lossless WebP |
| [15-inch MacBook Air](air15-m4.webp) | 1200×1200 | 960×960 | 1069.8 | 347.0 | 67.6% | Resize + lossless WebP |
| [Surface Laptop Go](surface-go.webp) | 2400×1350 | 960×540 | 765.5 | 107.1 | 86.0% | Resize + lossless WebP |
| [Beelink GTR9 Pro](beelink-gtr9.webp) | 1500×1500 | 960×960 | 849.9 | 223.2 | 73.7% | Resize + lossless WebP |
| [Dell XPS 15 (2023)](dell-xps.webp) | 1600×659 | 960×395 | 846.8 | 258.0 | 69.5% | Resize + lossless WebP |
| [24-inch iMac (M4)](imac-m4.webp) | 1200×1200 | 960×960 | 743.3 | 217.8 | 70.7% | Resize + lossless WebP |
| [13-inch MacBook Air](air13.webp) | 1200×1200 | 960×960 | 711.7 | 236.8 | 66.7% | Resize + lossless WebP |
| [HP Z2 Mini G1a](hp-z2a.webp) | 2291×1079 | 960×452 | 617.9 | 157.1 | 74.6% | Resize + lossless WebP |
| [MacBook Air (M1)](air13-m1.webp) | 1200×1200 | 960×960 | 650.5 | 213.6 | 67.2% | Resize + lossless WebP |
| [13-inch MacBook Pro (M1 / M2)](pro13-m1.webp) | 1200×1200 | 960×960 | 728.8 | 298.2 | 59.1% | Resize + lossless WebP |
| [16-inch MacBook Pro](book16.webp) | 1200×1200 | 960×960 | 500.1 | 183.5 | 63.3% | Resize + lossless WebP |
| [Dell G15 5530](dell-g15.webp) | 1600×774 | 1600×774 | 769.2 | 484.9 | 37.0% | Lossless WebP |
| [14-inch MacBook Pro](book14.webp) | 1200×1200 | 960×960 | 387.8 | 138.1 | 64.4% | Resize + lossless WebP |
| [Mac mini (M1 / M2)](mini-m1.webp) | 1200×1200 | 960×960 | 335.4 | 115.0 | 65.7% | Resize + lossless WebP |
| [NVIDIA DGX A100](dgx-a100.webp) | 639×325 | 639×325 | 427.5 | 217.8 | 49.1% | Lossless WebP |
| [RX 580](rx580.webp) | 800×500 | 800×500 | 426.4 | 261.6 | 38.7% | Lossless WebP |
| [GeForce RTX 5090](rtx5090.webp) | 800×800 | 800×800 | 345.1 | 218.2 | 36.8% | Lossless WebP |
| [RTX 4090](rtx4090.webp) | 800×800 | 800×800 | 262.9 | 156.5 | 40.5% | Lossless WebP |
| [MINISFORUM MS-S1 MAX](minisforum-ms-s1.webp) | 1600×1600 | 960×960 | 384.3 | 279.0 | 27.4% | Resize + lossless WebP |
| [RTX 3060](rtx3060.webp) | 800×800 | 800×800 | 245.4 | 148.1 | 39.6% | Lossless WebP |
| [ASUS ROG Zephyrus G14 (2024)](rog-g14.webp) | 2020×1408 | 960×669 | 166.5 | 81.8 | 50.9% | Resize + lossless WebP |
| [ThinkPad T14 Gen 4](thinkpad-t14.webp) | 584×584 | 584×584 | 166.5 | 103.9 | 37.6% | Lossless WebP |
| [HP ZBook Ultra G1a 14](hp-zbook.webp) | 1276×996 | 1276×996 | 138.3 | 113.0 | 18.3% | Lossless WebP |
| [RTX 3090](rtx3090.webp) | 1000×1000 | 1000×1000 | 157.2 | 133.9 | 14.8% | Lossless WebP |
| [HP EliteBook 840 G10](hp-elitebook-co.webp) | 1200×901 | 1200×901 | 191.7 | 171.2 | 10.7% | Lossless WebP |
| [RTX 3070](rtx3070.webp) | 1000×1000 | 1000×1000 | 191.9 | 172.2 | 10.3% | Lossless WebP |
| [RTX 4070 SUPER](rtx4070super.webp) | 800×800 | 800×800 | 113.0 | 93.8 | 17.0% | Lossless WebP |
| [RTX 5060](rtx5060.webp) | 800×800 | 800×800 | 119.9 | 101.2 | 15.6% | Lossless WebP |
| [Mac Studio](studio.webp) | 744×302 | 744×302 | 54.6 | 35.9 | 34.1% | Lossless WebP |
| [NVIDIA GeForce RTX 5080](rtx5080.webp) | 800×800 | 800×800 | 89.9 | 71.6 | 20.4% | Lossless WebP |
| [RTX 5070 Ti](rtx5070ti.webp) | 800×800 | 800×800 | 114.1 | 95.9 | 16.0% | Lossless WebP |
| [NVIDIA GeForce RTX 5070](rtx5070.webp) | 800×800 | 800×800 | 92.1 | 78.2 | 15.1% | Lossless WebP |
| [NVIDIA DGX Spark](dgx-spark.webp) | 800×800 | 800×800 | 57.5 | 46.6 | 18.9% | Lossless WebP |
| [Mac mini (2024)](mini.webp) | 744×302 | 744×302 | 21.3 | 12.7 | 40.3% | Lossless WebP |
| [HP Envy 17-cw](hp-envy.webp) | 500×376 | 500×376 | 41.3 | 35.7 | 13.5% | Lossless WebP |
| [HP Pavilion Plus 14-ey](hp-pavilion.webp) | 575×432 | 575×432 | 38.9 | 33.8 | 13.1% | Lossless WebP |
| [ASUS ROG Flow Z13 (2025)](rog-z13.webp) | 1000×808 | 1000×808 | 28.6 | 26.3 | 8.1% | Lossless WebP |
| [Radeon RX 6800 XT / 6900 XT](rx6800xt.webp) | 900×500 | 900×500 | 32.1 | 32.1 | 0.0% | Keep original |
| [Radeon RX 7900 XT / XTX](rx7900.webp) | 900×500 | 900×500 | 76.3 | 76.3 | 0.0% | Keep original |
| [RX 9060 XT](rx9060xt.webp) | 900×500 | 900×500 | 22.5 | 22.5 | 0.0% | Keep original |
| [Surface Laptop Studio](surface-studio.webp) | 2240×1619 | 2240×1619 | 154.8 | 154.8 | 0.0% | Keep original |

Per-file byte counts, hashes and comparison metrics are retained in [optimization-audit.json](optimization-audit.json). Original source URLs remain in [inventory.json](inventory.json).

Validation: 44/44 unique transparent images passed the asset audit; 114 focused hardware tests passed; production Electron/Vite build passed and all 44 emitted files match the optimized asset hashes. Browser inspection confirmed every gallery photo loaded with no horizontal overflow.
