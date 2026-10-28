"""Minimal standard-library readers for the FLDOE school-grades spreadsheets.

The project environment has neither ``xlrd`` nor ``openpyxl``, so this module reads the
two formats FLDOE uses directly:

* ``.xls`` (2013-14 to 2020-21): BIFF8 records inside an OLE2 compound file;
* ``.xlsx`` (2021-22 onward): Office Open XML in a zip archive.

Both return the first worksheet as a list of rows, each a list of cell values (``str``,
``float`` or ``None``). Only the record types these files use are handled: shared and
inline strings, numbers, RK / MULRK numbers, formula results and booleans. Formatting,
dates and merged cells are ignored.
"""
from __future__ import annotations

import struct
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

Row = list  # list[str | float | None]


# ----------------------------------------------------------------------------- .xls

def _ole_stream(data: bytes, names=("Workbook", "Book")) -> bytes:
    """Return the workbook stream of an OLE2 compound file (regular sectors only)."""
    if data[:8] != b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        raise ValueError("not an OLE2 compound file")
    ssz = 1 << struct.unpack_from("<H", data, 30)[0]
    nfat, dir_start = struct.unpack_from("<II", data, 44)
    mini_cutoff = struct.unpack_from("<I", data, 56)[0]
    difat_start, n_difat = struct.unpack_from("<II", data, 68)
    difat = list(struct.unpack_from("<109I", data, 76))
    per = ssz // 4

    def sector(i: int) -> bytes:
        return data[512 + i * ssz: 512 + (i + 1) * ssz]

    s = difat_start
    for _ in range(n_difat):
        vals = struct.unpack(f"<{per}I", sector(s))
        difat += vals[:-1]
        s = vals[-1]
    fat: list[int] = []
    for i in difat[:nfat]:
        fat += struct.unpack(f"<{per}I", sector(i))

    def chain(start: int) -> bytes:
        out, s, seen = [], start, set()
        while s < 0xFFFFFFFA:
            if s in seen:
                raise ValueError("cyclic sector chain")
            seen.add(s)
            out.append(sector(s))
            s = fat[s]
        return b"".join(out)

    dirs = chain(dir_start)
    for k in range(len(dirs) // 128):
        e = dirs[k * 128:(k + 1) * 128]
        nlen = struct.unpack_from("<H", e, 64)[0]
        name = e[:max(nlen - 2, 0)].decode("utf-16-le", "replace")
        start, size = struct.unpack_from("<II", e, 116)
        if e[66] == 2 and name in names:
            if size < mini_cutoff:
                raise ValueError("workbook stream in the mini stream is not supported")
            return chain(start)[:size]
    raise ValueError("no Workbook stream")


def _rk(v: int) -> float:
    if v & 2:
        x = float(struct.unpack("<i", struct.pack("<I", v))[0] >> 2)
    else:
        x = struct.unpack("<d", struct.pack("<Q", (v & 0xFFFFFFFC) << 32))[0]
    return x / 100 if v & 1 else x


def _xl_string(buf: bytes, p: int, len_bytes: int = 2) -> tuple[str, int]:
    """An unsplit BIFF8 unicode string (LABEL, STRING, BOUNDSHEET name)."""
    n = buf[p] if len_bytes == 1 else struct.unpack_from("<H", buf, p)[0]
    p += len_bytes
    flags = buf[p]
    p += 1
    if flags & 8:
        p += 2
    if flags & 4:
        p += 4
    if flags & 1:
        return buf[p:p + 2 * n].decode("utf-16-le"), p + 2 * n
    return buf[p:p + n].decode("latin-1"), p + n


def _parse_sst(chunks: list[bytes]) -> list[str]:
    """Shared-string table, following strings split across CONTINUE records."""
    ci, buf, p = 0, chunks[0], 8
    unique = struct.unpack_from("<I", buf, 4)[0]
    out = []
    for _ in range(unique):
        if p >= len(buf):
            ci, p = ci + 1, 0
            buf = chunks[ci]
        n = struct.unpack_from("<H", buf, p)[0]
        flags = buf[p + 2]
        p += 3
        nruns = ext = 0
        if flags & 8:
            nruns = struct.unpack_from("<H", buf, p)[0]
            p += 2
        if flags & 4:
            ext = struct.unpack_from("<I", buf, p)[0]
            p += 4
        wide, left, parts = flags & 1, n, []
        while True:
            avail = (len(buf) - p) // (2 if wide else 1)
            take = min(left, avail)
            if wide:
                parts.append(buf[p:p + 2 * take].decode("utf-16-le"))
                p += 2 * take
            else:
                parts.append(buf[p:p + take].decode("latin-1"))
                p += take
            left -= take
            if left == 0:
                break
            ci += 1
            buf = chunks[ci]
            wide, p = buf[0] & 1, 1
        out.append("".join(parts))
        skip = 4 * nruns + ext
        while skip > 0:
            if skip <= len(buf) - p:
                p += skip
                skip = 0
            else:
                skip -= len(buf) - p
                ci, p = ci + 1, 0
                buf = chunks[ci]
    return out


def _xls_records(path: Path) -> list:
    wb = _ole_stream(Path(path).read_bytes())
    recs = []
    pos = 0
    while pos + 4 <= len(wb):
        rt, ln = struct.unpack_from("<HH", wb, pos)
        recs.append((rt, pos, wb[pos + 4:pos + 4 + ln]))
        pos += 4 + ln
    return recs


def _xls_globals(recs: list) -> tuple[list, list, list]:
    sheets, names, sst = [], [], []
    for i, (rt, _, d) in enumerate(recs):
        if rt == 0x0085:                                  # BOUNDSHEET
            sheets.append(struct.unpack_from("<I", d, 0)[0])
            names.append(_xl_string(d, 6, len_bytes=1)[0])
        elif rt == 0x00FC:                                # SST (+ CONTINUE)
            chunks = [d]
            j = i + 1
            while j < len(recs) and recs[j][0] == 0x003C:
                chunks.append(recs[j][2])
                j += 1
            sst = _parse_sst(chunks)
        elif rt == 0x000A:                                # EOF of the globals
            break
    return sheets, names, sst


def read_xls(path: Path, sheet: int = 0) -> list[Row]:
    recs = _xls_records(path)
    sheets, _, sst = _xls_globals(recs)
    start = sheets[sheet]
    cells: dict[tuple[int, int], object] = {}
    pending = None
    for rt, off, d in recs:
        if off < start:
            continue
        if rt == 0x000A:
            break
        if rt == 0x00FD:                                  # LABELSST
            r, c, _, k = struct.unpack_from("<HHHI", d)
            cells[r, c] = sst[k]
        elif rt == 0x0203:                                # NUMBER
            r, c, _, x = struct.unpack_from("<HHHd", d)
            cells[r, c] = x
        elif rt == 0x027E:                                # RK
            r, c, _, v = struct.unpack_from("<HHHI", d)
            cells[r, c] = _rk(v)
        elif rt == 0x00BD:                                # MULRK
            r, c0 = struct.unpack_from("<HH", d)
            for k in range((len(d) - 6) // 6):
                cells[r, c0 + k] = _rk(struct.unpack_from("<I", d, 4 + 6 * k + 2)[0])
        elif rt == 0x0204:                                # LABEL
            r, c, _ = struct.unpack_from("<HHH", d)
            cells[r, c] = _xl_string(d, 6)[0]
        elif rt == 0x0205:                                # BOOLERR
            r, c, _, v, err = struct.unpack_from("<HHHBB", d)
            cells[r, c] = None if err else float(v)
        elif rt == 0x0006:                                # FORMULA
            r, c, _ = struct.unpack_from("<HHH", d)
            if d[12:14] == b"\xff\xff":
                kind = d[6]
                cells[r, c] = float(d[8]) if kind == 1 else None
                pending = (r, c) if kind == 0 else None
            else:
                cells[r, c] = struct.unpack_from("<d", d, 6)[0]
        elif rt == 0x0207 and pending is not None:        # STRING (formula result)
            cells[pending] = _xl_string(d, 0)[0]
            pending = None
    return _grid(cells)


# ---------------------------------------------------------------------------- .xlsx

_M = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def _col(ref: str) -> int:
    n = 0
    for ch in ref:
        if ch.isalpha():
            n = 26 * n + ord(ch.upper()) - 64
        else:
            break
    return n - 1


def _xlsx_sheets(z: zipfile.ZipFile) -> list[tuple[str, str]]:
    """(name, part path) of each worksheet, in workbook order."""
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    target = {r.get("Id"): r.get("Target") for r in rels}
    out = []
    for sh in wb.find(f"{_M}sheets"):
        t = target[sh.get(f"{_R}id")].lstrip("/")
        out.append((sh.get("name"), t if t.startswith("xl/") else f"xl/{t}"))
    return out


def read_xlsx(path: Path, sheet: int = 0) -> list[Row]:
    with zipfile.ZipFile(path) as z:
        part = _xlsx_sheets(z)[sheet][1]
        sst = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")):
                sst.append("".join(x.text or "" for x in si.iter(f"{_M}t")
                                   if x not in si.findall(f".//{_M}rPh/{_M}t")))
        cells: dict[tuple[int, int], object] = {}
        with z.open(part) as fh:
            for _, el in ET.iterparse(fh):
                if el.tag != f"{_M}row":
                    continue
                r = int(el.get("r")) - 1
                for c in el.findall(f"{_M}c"):
                    kind, v = c.get("t"), c.find(f"{_M}v")
                    col = _col(c.get("r"))
                    if kind == "inlineStr":
                        val = "".join(x.text or "" for x in c.iter(f"{_M}t"))
                    elif v is None or v.text is None:
                        continue
                    elif kind == "s":
                        val = sst[int(v.text)]
                    elif kind in ("str", "e"):
                        val = v.text
                    elif kind == "b":
                        val = float(v.text)
                    else:
                        val = float(v.text)
                    cells[r, col] = val
                el.clear()
    return _grid(cells)


def _grid(cells: dict) -> list[Row]:
    if not cells:
        return []
    nr = max(r for r, _ in cells) + 1
    nc = max(c for _, c in cells) + 1
    out = [[None] * nc for _ in range(nr)]
    for (r, c), v in cells.items():
        out[r][c] = v
    return out


def sheet_names(path: Path) -> list[str]:
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        with zipfile.ZipFile(path) as z:
            return [n for n, _ in _xlsx_sheets(z)]
    return _xls_globals(_xls_records(path))[1]


def read(path: Path, sheet: int = 0) -> list[Row]:
    path = Path(path)
    return read_xlsx(path, sheet) if path.suffix.lower() == ".xlsx" else read_xls(path, sheet)
