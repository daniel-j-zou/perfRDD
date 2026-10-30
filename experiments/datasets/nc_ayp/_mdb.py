"""Minimal standard-library reader for Microsoft Access Jet 4 (.mdb) tables.

The NC School Report Card research databases of 2006-07 to 2011-12 are Access 2000
files. The project environment has no ``mdbtools``, ``pyodbc`` or ``access_parser``, so
this module reads the Jet 4 page format directly, following the layout documented by
mdbtools (HACKING) and Jackcess:

* 4096-byte pages. Page 2 is the table definition (TDEF) of the catalog ``MSysObjects``.
* A TDEF page holds the row count, the column descriptors (25 bytes each) and the
  column names (UCS-2); a definition can continue on further pages.
* Data pages (type 0x01) name their owning TDEF page at byte 4 and hold a row-offset
  table at byte 14. Offsets flagged 0x8000 are deleted rows, 0x4000 overflow pointers.
* A row starts with its column count; fixed-length columns follow at their offsets, then
  the variable-length data. The row ends with the variable-column offset table (stored
  backwards), the variable-column count and the null bitmap.

Only what the SRC databases use is decoded: booleans, integers, floats and doubles,
money, Jet 4 NUMERIC (decimal), dates (as float days since 1899-12-30), text (with
Jet 4 Unicode compression) and memo fields. Indexes are ignored; tables are read by
scanning every page of the file for data pages owned by the table.
"""
from __future__ import annotations

import mmap
import struct
from decimal import Decimal
from pathlib import Path

PAGE = 4096

# Column types
BOOL, BYTE, INT, LONG, MONEY, FLOAT, DOUBLE, DATETIME = 1, 2, 3, 4, 5, 6, 7, 8
BINARY, TEXT, OLE, MEMO, GUID, NUMERIC = 9, 10, 11, 12, 15, 16
_FIXED = {BYTE: "<B", INT: "<h", LONG: "<i", FLOAT: "<f", DOUBLE: "<d", DATETIME: "<d",
          MONEY: "<q"}


class Column:
    __slots__ = ("name", "type", "num", "var_num", "fixed_off", "size", "flags", "scale",
                 "compressed")

    def __repr__(self) -> str:
        return f"Column({self.name!r}, type={self.type}, size={self.size})"

    @property
    def fixed(self) -> bool:
        return bool(self.flags & 0x01)


def _text(b: bytes) -> str:
    """Jet 4 text: UCS-2LE, or 'compressed' when it starts with FF FE (one byte per
    character; a 0x00 byte toggles between one-byte and two-byte mode)."""
    if len(b) >= 2 and b[0] == 0xFF and b[1] == 0xFE:
        out, i, comp = [], 2, True
        while i < len(b):
            if b[i] == 0:
                comp = not comp
                i += 1
            elif comp:
                out.append(chr(b[i]))
                i += 1
            else:
                if i + 1 >= len(b):
                    break
                out.append(b[i:i + 2].decode("utf-16-le", "replace"))
                i += 2
        return "".join(out)
    return b.decode("utf-16-le", "replace")


def _numeric(b: bytes, scale: int) -> Decimal:
    """Jet 4 NUMERIC: a sign byte, then a 128-bit integer stored as four little-endian
    32-bit words, most significant word first."""
    words = struct.unpack_from("<4I", b, 1)
    v = (words[0] << 96) | (words[1] << 64) | (words[2] << 32) | words[3]
    if b[0] & 0x80:
        v = -v
    return Decimal(v).scaleb(-scale)


class MDB:
    """Read-only access to the user tables of a Jet 4 database."""

    def __init__(self, path):
        self.path = Path(path)
        self._fh = open(self.path, "rb")
        self.buf = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        if self.buf[0x14] != 1:
            raise ValueError(f"{path}: not a Jet 4 database (version byte "
                             f"{self.buf[0x14]})")
        self.n_pages = len(self.buf) // PAGE
        self._owners = None
        self._catalog = None

    def close(self) -> None:
        self.buf.close()
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ pages

    def page(self, n: int) -> bytes:
        return self.buf[n * PAGE:(n + 1) * PAGE]

    def _data_pages(self, tdef: int) -> list:
        if self._owners is None:
            owners: dict = {}
            b = self.buf
            for p in range(1, self.n_pages):
                o = p * PAGE
                if b[o] == 0x01 and b[o + 1] == 0x01:
                    owners.setdefault(struct.unpack_from("<I", b, o + 4)[0], []).append(p)
            self._owners = owners
        return self._owners.get(tdef, [])

    def _row_bounds(self, pg: bytes, i: int):
        """(start, end_exclusive, flags) of row i on a data page."""
        off = struct.unpack_from("<H", pg, 14 + 2 * i)[0]
        start = off & 0x1FFF
        end = PAGE if i == 0 else struct.unpack_from("<H", pg, 14 + 2 * (i - 1))[0] & 0x1FFF
        return start, end, off & 0xC000

    def _row_at(self, ptr: int) -> bytes:
        """Row bytes addressed by a 4-byte pointer (row number in the low byte, page in
        the upper three bytes)."""
        pg = self.page(ptr >> 8)
        s, e, _ = self._row_bounds(pg, ptr & 0xFF)
        return pg[s:e]

    # ------------------------------------------------------------------ TDEF

    def _tdef(self, tdef: int) -> bytes:
        pg = self.page(tdef)
        if pg[0] != 0x02:
            raise ValueError(f"page {tdef} is not a table definition")
        data = bytearray(pg)
        nxt = struct.unpack_from("<I", pg, 4)[0]
        while nxt:
            p = self.page(nxt)
            data += p[8:]
            nxt = struct.unpack_from("<I", p, 4)[0]
        return bytes(data)

    def columns(self, tdef: int):
        d = self._tdef(tdef)
        n_rows = struct.unpack_from("<I", d, 16)[0]
        n_cols = struct.unpack_from("<H", d, 45)[0]
        n_ridx = struct.unpack_from("<I", d, 51)[0]
        pos = 63 + 12 * n_ridx
        cols = []
        for _ in range(n_cols):
            c = Column()
            c.type = d[pos]
            c.num = struct.unpack_from("<H", d, pos + 5)[0]
            c.var_num = struct.unpack_from("<H", d, pos + 7)[0]
            c.scale = d[pos + 12]
            c.flags = d[pos + 15]
            c.compressed = bool(d[pos + 16] & 0x01)
            c.fixed_off = struct.unpack_from("<H", d, pos + 21)[0]
            c.size = struct.unpack_from("<H", d, pos + 23)[0]
            cols.append(c)
            pos += 25
        for c in cols:
            ln = struct.unpack_from("<H", d, pos)[0]
            c.name = d[pos + 2:pos + 2 + ln].decode("utf-16-le")
            pos += 2 + ln
        return cols, n_rows

    # ------------------------------------------------------------------ rows

    def _memo(self, b: bytes):
        if len(b) < 12:
            return None
        ln = struct.unpack_from("<I", b, 0)[0]
        flag, ln = ln & 0xC0000000, ln & 0x3FFFFFFF
        if flag == 0x80000000:
            return b[12:12 + ln]
        ptr = struct.unpack_from("<I", b, 4)[0]
        if flag == 0x40000000:
            return self._row_at(ptr)[:ln]
        out = bytearray()
        while ptr and len(out) < ln:
            r = self._row_at(ptr)
            ptr = struct.unpack_from("<I", r, 0)[0]
            out += r[4:]
        return bytes(out[:ln])

    def _value(self, c: Column, b: bytes):
        t = c.type
        if t in _FIXED:
            v = struct.unpack_from(_FIXED[t], b, 0)[0]
            return v / 10000 if t == MONEY else v
        if t == NUMERIC:
            return _numeric(b, c.scale)
        if t == TEXT:
            return _text(b)
        if t == MEMO:
            m = self._memo(b)
            return None if m is None else _text(m)
        if t == GUID:
            return b.hex()
        return bytes(b)

    def _crack(self, row: bytes, cols) -> dict:
        n = len(row)
        row_cols = struct.unpack_from("<H", row, 0)[0]
        nb = (row_cols + 7) // 8
        nullmask = row[n - nb:]
        n_var = struct.unpack_from("<H", row, n - nb - 2)[0]
        var_off = [struct.unpack_from("<H", row, n - nb - 4 - 2 * i)[0]
                   for i in range(n_var + 1)]
        n_fixed = row_cols - n_var
        out, fixed_seen, var_seen = {}, 0, 0
        for c in sorted(cols, key=lambda c: c.num):
            byte, bit = divmod(c.num, 8)
            present = byte < len(nullmask) and bool(nullmask[byte] & (1 << bit))
            if c.type == BOOL:
                out[c.name] = present
                continue
            if c.fixed and fixed_seen < n_fixed:
                fixed_seen += 1
                s = c.fixed_off + 2
                out[c.name] = self._value(c, row[s:s + c.size]) if present else None
            elif not c.fixed and var_seen < n_var:
                var_seen += 1
                s, e = var_off[c.var_num], var_off[c.var_num + 1]
                out[c.name] = self._value(c, row[s:e]) if present and e > s else None
            else:
                out[c.name] = None
        return out

    def rows(self, tdef: int, columns=None):
        """Yield every live row of the table defined at page ``tdef`` as a dict. With
        ``columns``, only those columns are decoded."""
        cols, _ = self.columns(tdef)
        if columns is not None:
            want = set(columns)
            cols = [c for c in cols if c.name in want]
        for p in self._data_pages(tdef):
            pg = self.page(p)
            n = struct.unpack_from("<H", pg, 12)[0]
            for i in range(n):
                s, e, flags = self._row_bounds(pg, i)
                if flags & 0x8000 or e <= s:
                    continue
                row = pg[s:e]
                if flags & 0x4000:
                    row = self._row_at(struct.unpack_from("<I", row, 0)[0])
                yield self._crack(row, cols)

    # ------------------------------------------------------------------ catalog

    def tables(self) -> dict:
        """User table name -> TDEF page."""
        if self._catalog is None:
            cat = {}
            for r in self.rows(2, ["Id", "Name", "Type", "Flags"]):
                if r["Type"] == 1 and not (r["Flags"] or 0) & 0x80000002:
                    cat[r["Name"]] = r["Id"] & 0x00FFFFFF
            self._catalog = cat
        return self._catalog

    def table(self, name: str, columns=None) -> list:
        """All rows of a user table (case-insensitive name) as a list of dicts."""
        cat = {k.lower(): v for k, v in self.tables().items()}
        return list(self.rows(cat[name.lower()], columns))

    def table_columns(self, name: str) -> list:
        cat = {k.lower(): v for k, v in self.tables().items()}
        return [c.name for c in self.columns(cat[name.lower()])[0]]

    def row_count(self, name: str) -> int:
        cat = {k.lower(): v for k, v in self.tables().items()}
        return self.columns(cat[name.lower()])[1]
