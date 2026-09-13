import csv
from pathlib import Path

from ..common import CSV_FIELDS


def open_output_csv(output_path):
    """Open the output CSV for appending, writing the header if it's new."""
    output_path = Path(output_path)
    is_new = not output_path.exists() or output_path.stat().st_size == 0
    f = open(output_path, 'a', newline='')
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
    if is_new:
        writer.writeheader()
        f.flush()
    return f, writer
