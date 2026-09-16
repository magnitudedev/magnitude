"""Generate the reviewable complete spec/coverage tables from the curated JSON."""
import json
from collections import Counter
from pathlib import Path
root = Path(__file__).resolve().parents[1]
facts = json.loads((root / 'facts.json').read_text())
coverage = json.loads((root / 'coverage.json').read_text())
inventory = json.loads((root / 'inventory.json').read_text())
assert sorted(row['photoId'] for row in coverage) == sorted(row['id'] for row in inventory)
counts = Counter(row['target'] for row in facts)
def field_count(target, label):
    return sum(row['target'] == target and any(f['label'] == label for f in row['facts']) for row in facts)
gpu_units = sum(row['target'] == 'Accelerator' and any(any(unit in f['label'].lower() for unit in ['cuda cores', 'compute units', 'xe cores', 'stream processors']) for f in row['facts']) for row in facts)
bandwidth_entries = [row for row in facts if row['target'] == 'Accelerator' and any(f['label'] == 'Memory bandwidth (spec)' for f in row['facts'] + [f for v in row.get('memoryVariants', []) for f in v['facts']])]
lines = ['# Hardware specification coverage', '',
         f"{len(facts)} catalog entries: {counts['Processor']} processors, {counts['Accelerator']} accelerators and {counts['Device']} device specifications. {len(coverage)} of {len(inventory)} photo groups have researched coverage records.", '',
         'Coverage means the listed factory chip configurations have facts or explicitly described configuration limits. It does not mean every computer on the market is recognized, every specification field is known, or every installed GPU is visible to the inference backend. Standalone GPU photos imply no CPU model.', '',
         'Observed memory and OS-reported CPU cores take precedence as descriptions of the installed configuration. Published values are nominal specifications. CPU core options are not installed counts; physical CPU counts may select a variant only when the published mapping is unique. GPU units from different vendors are not interchangeable.', '',
         f"CPU coverage: {field_count('Processor', 'CPU cores / processor (spec)')} fixed core counts and {field_count('Processor', 'CPU core options (spec)')} configurable core-count entries. GPU unit counts: {gpu_units}/{counts['Accelerator']} accelerator entries. Missing GPU counts are the ambiguous RTX 3050 Laptop GPU variants and the unresolved FirePro D500 specification. Generic integrated-graphics names and the optional T14 MX550 do not receive guessed core counts. Bandwidth is published only when the exact chip/configuration supports an unambiguous nominal value.", '',
         f"GPU bandwidth coverage: {len(bandwidth_entries)}/{counts['Accelerator']} accelerator entries, including capacity-dependent variants. Mobile and vendor-rated maxima use ‘Up to’; these are not measured transfer rates. Capacity variants require an observed dedicated physical memory total within 1% of one catalog size, allowing small driver reservations. Shared memory, free memory and allocation budgets never select a variant.", '',
         '## Photo group to processor and accelerator mapping', '', '| Photo group | Researched processors | Researched accelerators | Configuration limits |', '| --- | --- | --- | --- |']
for row in coverage:
    chips = ', '.join(row['processors']) or 'No CPU implied'
    accelerators = ', '.join(row['accelerators']) or 'Configuration-dependent; see limits'
    sources = ' '.join(f'[source {i+1}]({url})' for i, url in enumerate(row['sources']))
    lines.append(f"| {row['photoId']} | {chips} | {accelerators} | {row['configurationNote']} {sources} |")
lines += ['', '## Every published entry', '', '| Target / names | Published facts | Variant facts | Primary sources |', '| --- | --- | --- | --- |']
for row in sorted(facts, key=lambda row: (row['target'], row['names'][0])):
    values = '; '.join(f"{f['label']}: {f['value']}" for f in row['facts'])
    variants = '; '.join(f"{v['physicalCpuCores']} observed physical CPU cores → " + ', '.join(f"{f['label']}: {f['value']}" for f in v['facts']) for v in row.get('variants', []))
    memory_variants = '; '.join(f"{v['memoryGiB']} GB dedicated VRAM → " + ', '.join(f"{f['label']}: {f['value']}" for f in v['facts']) for v in row.get('memoryVariants', []))
    variants = '; '.join(filter(None, [variants, memory_variants])) or '—'
    sources = ' '.join(f'[source {i+1}]({url})' for i,url in enumerate([row['source'], *row.get('additionalSources', [])]))
    lines.append(f"| {row['target']}: {', '.join(row['names'])} | {values} | {variants} | {sources} |")
(root / 'FACTS.md').write_text('\n'.join(lines) + '\n')
print(dict(counts), 'photo groups:', len(coverage), 'JSON bytes:', (root/'facts.json').stat().st_size)
