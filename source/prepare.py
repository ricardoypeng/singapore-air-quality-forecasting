"""Rebuild current model inputs from the versioned raw observations, without training."""
import argparse
import hashlib
import json
from pathlib import Path
from source.prepare_data import prepare_data
from source.prepare_weather_data import prepare

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir', type=Path, default=ROOT)
    parser.add_argument('--verify-only', action='store_true', help='Check raw source hashes without rebuilding')
    args = parser.parse_args()
    root = args.project_dir.resolve()
    provenance = json.loads((root / 'data/raw_sources.json').read_text())
    for name, expected in provenance['sha256'].items():
        path = root / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Raw source checksum mismatch: {name}')
    print(f"Verified {len(provenance['sha256'])} raw source files.", flush=True)
    if args.verify_only:
        return
    output = root / 'tmp/_project/prepared-data/weather_2024'
    if output.exists():
        raise FileExistsError('Prepared output already exists; use a fresh checkout or archive that output first')
    pm = root / 'tmp/_project/data-preparation/pm25'
    prepare_data(root / 'data/singapore-pm25-hourly-raw.csv', pm)
    metadata = prepare(argparse.Namespace(raw_dir=root / 'data/weather_raw_s107', pm_dir=pm,
        output_dir=output, station='S107', start_year=2017, end_year=2024, overwrite=False))
    expected = provenance['expected_windows']
    for split, count in expected.items():
        actual = metadata['window_starts']['files'][f'{split}_windows_24h.npz']['count']
        if actual != count:
            raise ValueError(f'{split}: expected {count} windows, got {actual}')
    print('Prepared data rebuilt and window counts verified; no model training performed.')


if __name__ == '__main__':
    main()
