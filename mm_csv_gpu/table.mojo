"""`GpuCsvTable` -- one document, for callers who do not want a scanner.

This is a `GpuCsvScanner` that has been handed exactly one document. It is the
right type when a program reads a CSV file and exits, and the wrong one when it
reads many: every table allocates its own buffers, and on a large document that
allocation costs an order of magnitude more than the scan. Two documents means
`GpuCsvScanner`.
"""

from max.gpu.host import DeviceContext
from std.pathlib import Path

from .kernels import COMMA
from .scanner import GpuCsvScanner


struct GpuCsvTable[separator: UInt8 = COMMA](Movable, Sized):
    """A CSV document whose fields were found on the GPU.

    Parameters:
        separator: The byte between fields. `,` by default; `\\t` for TSV.
    """

    var _scanner: GpuCsvScanner[Self.separator]

    var column_count: Int
    """Fields in the first row, which RFC 4180 says every row must match."""

    def __init__(out self, ctx: DeviceContext, path: Path) raises:
        """Reads `path` and finds every field in it.

        Args:
            ctx: The device to run on.
            path: The document to read.

        Raises:
            Error: If the file cannot be read, is 2 GiB or larger, or a device
                allocation or launch fails.
        """
        self._scanner = GpuCsvScanner[Self.separator](ctx)
        self._scanner.scan(ctx, path)
        self.column_count = self._scanner.column_count

    def __len__(self) -> Int:
        """Returns the number of fields the document holds.

        Returns:
            Every field in every row, including empty ones.
        """
        return len(self._scanner)

    def row_count(self) -> Int:
        """Returns the number of rows.

        Returns:
            Fields divided by columns, or 0 for an empty document.
        """
        return self._scanner.row_count()

    def is_ragged(self) -> Bool:
        """Returns whether some row has a different field count from the first.

        Returns:
            True if the field count is not a whole number of rows.
        """
        return self._scanner.is_ragged()

    def field[
        origin: ImmOrigin, //
    ](ref[origin] self, row: Int, column: Int) raises -> StringSlice[origin]:
        """Returns a field exactly as it appears in the document.

        Parameters:
            origin: The origin of the borrow.

        Args:
            row: Zero-based row index.
            column: Zero-based column index.

        Raises:
            Error: If `row` or `column` is out of range.

        Returns:
            The bytes between the delimiters, borrowed.
        """
        # The slice borrows the scanner, which borrows `self`; the two
        # origins are the same memory but the compiler names them
        # differently, so this says so.
        return rebind[StringSlice[origin]](self._scanner.field(row, column))
