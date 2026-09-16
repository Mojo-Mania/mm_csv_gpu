"""Reading a CSV document with the GPU."""

from max.gpu.host import DeviceContext
from mm_csv_gpu import GpuCsvTable
from std.pathlib import Path

comptime SAMPLE = "/tmp/mm_csv_gpu_example.csv"


def main() raises:
    with open(SAMPLE, "w") as f:
        f.write(
            "name,note,score\r\n"
            'Ada,"first, and foremost",100\r\n'
            'Grace,"said ""hello""",99\r\n'
        )

    var ctx = DeviceContext()
    var table = GpuCsvTable(ctx, Path(SAMPLE))
    print("rows:", table.row_count(), " columns:", table.column_count)
    print("ragged?", table.is_ragged())

    for row in range(1, table.row_count()):
        print(
            "  ",
            table.field(row, 0),
            "|",
            table.field(row, 1),
            "|",
            table.field(row, 2),
        )

    # Fields come back raw: the quotes are still on them, and a doubled quote
    # inside is still doubled. Undoing that is mm_csv's job, not this one.
    try:
        _ = table.field(0, 99)
    except error:
        print("  out of range raises:", error)
