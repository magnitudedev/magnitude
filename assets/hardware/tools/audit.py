"""Validate bundled photos and regenerate the complete mapping. Requires Pillow.
Run: python assets/hardware/tools/audit.py
Only reads source images; never modifies photography.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path
from PIL import Image

root = Path(__file__).resolve().parents[1]
entries = json.loads((root / 'inventory.json').read_text())
seen = set()
rows = []
failures = []
for entry in entries:
    path = root / entry['file']
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest not in seen, f'Duplicate image: {path.name}'
    seen.add(digest)
    with Image.open(path) as image:
        alpha = image.convert('RGBA').getchannel('A')
        transparent = alpha.getextrema() == (0, 255)
        bounds = alpha.getbbox()
        assert bounds, f'Empty image: {path.name}'
        x, y, right, bottom = bounds
        assert entry['framing'] == dict(sourceWidth=image.width, sourceHeight=image.height, x=x, y=y, width=right-x, height=bottom-y), f'Stale framing: {path.name}'
        if not transparent:
            failures.append(f'{path.name}: no transparent background')
        assert max(image.size) >= 500, f'Insufficient source resolution: {path.name}'
        rows.append({ 'id': entry['id'], 'file': path.name, 'width': image.width, 'height': image.height,
                     'bytes': path.stat().st_size, 'transparent': transparent, 'sha256': digest })
assert set(p.name for p in root.iterdir() if p.suffix in ('.jpg', '.png', '.webp')) == set(e['file'] for e in entries)
(root / 'photo-audit.json').write_text(json.dumps(rows, indent=2) + '\n')
counts = Counter(e['category'] for e in entries)
lines = ['# Full hardware photo mapping', '', f'{len(entries)} distinct bundled photos. Each row is one shared-photo group. Matching ignores case and collapses whitespace; every other identifier detail is significant.', '',
         'Enclosure matches take priority. Component photos require observed dedicated memory and are suppressed for known portable and all-in-one systems. Unknown products retain their observed name and firmware category without an invented enclosure photo. A GPU photo represents its chip/card family; it does not identify the installed board partner, cooler, or color.', '',
         ' | Category | Photos |', ' | --- | ---: |']
lines += [f' | {category} | {count} |' for category, count in sorted(counts.items())]
lines += ['', '## Every asset and exact match', '', '| Photo / shared group | Category | Required manufacturer | Field and accepted identifiers | Additional constraint |', '| --- | --- | --- | --- | --- |']
for entry in entries:
    match = entry['match']
    vendor = 'Apple; Apple Inc.' if match['_tag'] == 'Apple' else '; '.join(match.get('manufacturers', [])) if match['_tag'] == 'Pc' else 'GPU vendor prefix optional'
    field = 'accelerator name' if match['_tag'] == 'Gpu' else match.get('field', 'model')
    names = '; '.join(f'`{name}`' for name in match.get('models', match.get('names', [])))
    constraint = 'Processor token: ' + '; '.join(entry['processorModels']) if entry.get('processorModels') else '—'
    lines.append(f"| [{entry['subject']}]({entry['file']}) (`{entry['id']}`) | {entry['category']} | {vendor} | {field}: {names} | {constraint} |")
lines += ['', '## Source and quality records', '', 'The machine-readable [inventory](inventory.json) retains every product page, original image URL, and additional firmware-identity evidence. [Photo audit](photo-audit.json) records dimensions, real transparency and SHA-256 hashes. Run `python assets/hardware/tools/audit.py` with Pillow to regenerate this document and validate every asset.', '', 'GPU matching removes only the NVIDIA / AMD / Intel and GeForce / Radeon prefixes. Laptop GPU, Ti, SUPER, XT, XTX and regional suffixes remain significant. Lenovo product-version matching never falls back to a similarly named family. The reused Dell XPS 15 9530 identifier additionally requires a verified 2023 processor token.', '']
(root / 'MAPPING.md').write_text('\n'.join(lines))
print(f'{len(entries)} distinct photos; {sum(row["transparent"] for row in rows)} transparent; {sum((root / e["file"]).stat().st_size for e in entries):,} image bytes')
if failures:
    raise SystemExit('\n'.join(failures))
