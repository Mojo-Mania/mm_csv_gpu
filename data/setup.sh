#!/usr/bin/env bash
# Builds the CSV files the benchmark reads. None of them is committed -- they
# are about 550 MB together.
#
#   bash data/setup.sh
#
# `small.csv` is byte for byte mm_csv's `no_escaping.csv`: same generator,
# same seed, same columns. That is deliberate, so a number measured here can
# be put next to a number measured there without an argument about the input.
# The other two are the sizes a GPU is actually for.
#
#   small.csv     23 MB    255 360 rows    8 cols    0% quoted
#   large.csv    253 MB  2 800 000 rows    8 cols    0% quoted
#   quoted.csv   272 MB  2 200 000 rows   10 cols   10% quoted
#
# All use CRLF, as RFC 4180 requires.
set -eu
cd "$(dirname "$0")"

echo "generating (a minute or so)"
python3 generate.py

for f in small.csv large.csv quoted.csv; do
  python3 - "$f" <<'PY'
import os, sys
size = os.path.getsize(sys.argv[1])
with open(sys.argv[1], "rb") as handle:
    head = handle.read(1 << 20)
rows = head.count(b"\r\n")
print("  %-12s %12d bytes  ~%d rows" % (
    sys.argv[1], size, rows * size // len(head)))
PY
done
