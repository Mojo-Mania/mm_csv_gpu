"""Generates the benchmark CSVs, matching the shapes in setup.sh.

The same generator, seed and column layout as mm_csv's `no_escaping.csv`, so
a GPU number here is comparable with a CPU number there. `small.csv` is that
document exactly; `large.csv` is eleven times as many rows, because a GPU is
not the interesting choice at 23 MB.

Deterministic: the same seed gives the same bytes on every machine.
"""
import random

random.seed(20260915)

WORDS = ("Balance Payments international investment position Quarterly major "
         "components Actual Dollars Northland region Female Male Total "
         "Regional council years quantile measure area geography ethnic "
         "value Suppressed STATUS UNITS MAGNTUDE Subject Group Series").split()


def field(rng, long_tail):
    """A field whose length distribution matches the originals."""
    roll = rng.random()
    if roll < 0.45:
        return format(rng.random() * 1000, ".2f")
    if roll < 0.75:
        return rng.choice(WORDS)
    if roll < 0.95:
        return " ".join(rng.choice(WORDS) for _ in range(2))
    return " ".join(rng.choice(WORDS) for _ in range(long_tail))


def write(path, rows, header, quoted_fraction):
    rng = random.Random(20260915)
    with open(path, "w", newline="") as handle:
        handle.write(",".join(header) + "\r\n")
        for _ in range(rows):
            cells = []
            for index in range(len(header)):
                value = field(rng, 6)
                # One field per row carries a comma, which is what forces the
                # quoting in the second file.
                if quoted_fraction and index == len(header) - 2:
                    value = '"%s, %s"' % (rng.choice(WORDS), value)
                cells.append(value)
            handle.write(",".join(cells) + "\r\n")


COLUMNS = ["measure", "quantile", "area", "sex", "age", "geography",
           "ethnic", "value"]

write("small.csv", 255_360, COLUMNS, 0.0)
write("large.csv", 2_800_000, COLUMNS, 0.0)
write("quoted.csv", 2_200_000,
      ["Series_reference", "Period", "Data_value", "Suppressed", "STATUS",
       "UNITS", "MAGNTUDE", "Subject", "Group", "Series_title_1"], 0.1)
