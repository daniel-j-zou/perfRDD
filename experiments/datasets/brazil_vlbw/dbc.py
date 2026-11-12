"""Pure-Python reader for DATASUS .dbc files (DBF with PKWARE DCL "implode" compression).

Header size is at bytes 8-9 (little endian); the compressed stream starts at
header_size + 4. Decoder follows zlib's contrib/blast (public algorithm).
Run files through this with `python -I`; it only reads bytes, never executes them.
"""
import struct

_LITLEN = [11,124,8,7,28,7,188,13,76,4,10,8,12,10,12,10,8,23,8,9,7,6,7,8,7,6,55,8,23,24,12,11,7,9,11,12,6,7,22,5,7,24,6,11,9,6,7,22,7,11,38,7,9,8,25,11,8,11,9,12,8,12,5,38,5,38,5,11,7,5,6,21,6,10,53,8,7,24,10,27,44,253,253,253,252,252,252,13,12,45,12,45,12,61,12,45,44,173]
_LENLEN = [2,35,36,53,38,23]
_DISTLEN = [2,20,53,230,247,151,248]
_BASE = [3,2,4,5,6,7,8,9,10,12,16,24,40,72,136,264]
_EXTRA = [0,0,0,0,0,0,0,0,1,2,3,4,5,6,7,8]
MAXBITS = 13


def _construct(rep):
    lengths = []
    for b in rep:
        left = (b >> 4) + 1
        ln = b & 15
        lengths.extend([ln] * left)
    count = [0] * (MAXBITS + 1)
    for ln in lengths:
        count[ln] += 1
    offs = [0] * (MAXBITS + 1)
    for ln in range(1, MAXBITS):
        offs[ln + 1] = offs[ln] + count[ln]
    symbol = [0] * len(lengths)
    for s, ln in enumerate(lengths):
        if ln:
            symbol[offs[ln]] = s
            offs[ln] += 1
    return count, symbol


_LIT = _construct(_LITLEN)
_LEN = _construct(_LENLEN)
_DIST = _construct(_DISTLEN)


def explode(data):
    pos = 0
    bitbuf = 0
    bitcnt = 0
    n = len(data)

    def bits(need):
        nonlocal pos, bitbuf, bitcnt
        val = bitbuf
        while bitcnt < need:
            val |= data[pos] << bitcnt
            pos += 1
            bitcnt += 8
        bitbuf = val >> need
        bitcnt -= need
        return val & ((1 << need) - 1)

    def decode(h):
        count, symbol = h
        code = first = index = 0
        for ln in range(1, MAXBITS + 1):
            code |= bits(1) ^ 1
            c = count[ln]
            if code < first + c:
                return symbol[index + (code - first)]
            index += c
            first += c
            first <<= 1
            code <<= 1
        raise ValueError("bad code")

    lit = bits(8)
    dic = bits(8)
    if lit > 1 or dic < 4 or dic > 6:
        raise ValueError("not an implode stream")
    out = bytearray()
    while True:
        if bits(1):
            s = decode(_LEN)
            ln = _BASE[s] + bits(_EXTRA[s])
            if ln == 519:
                break
            sh = 2 if ln == 2 else dic
            dist = decode(_DIST) << sh
            dist += bits(sh)
            dist += 1
            if dist > len(out):
                raise ValueError("distance too far")
            if dist >= ln:
                st = len(out) - dist
                out += out[st:st + ln]
            else:
                for _ in range(ln):
                    out.append(out[-dist])
        else:
            out.append(decode(_LIT) if lit else bits(8))
    return bytes(out)


def read_dbc(path):
    """Return (list of field names, list of (name,type,len), list of record byte strings)."""
    raw = open(path, "rb").read()
    hsize = struct.unpack("<H", raw[8:10])[0]
    rsize = struct.unpack("<H", raw[10:12])[0]
    nrec = struct.unpack("<I", raw[4:8])[0]
    body = explode(raw[hsize + 4:])
    hdr = raw[:hsize]
    fields = []
    i = 32
    while i < hsize and hdr[i] != 0x0D:
        name = hdr[i:i + 11].split(b"\0")[0].decode("latin-1")
        ftype = chr(hdr[i + 11])
        flen = hdr[i + 16]
        fields.append((name, ftype, flen))
        i += 32
    assert sum(f[2] for f in fields) + 1 == rsize, (sum(f[2] for f in fields) + 1, rsize)
    if len(body) < nrec * rsize:
        raise ValueError(f"short body {len(body)} < {nrec*rsize}")
    return fields, nrec, rsize, body


def to_frame(path):
    import pandas as pd
    fields, nrec, rsize, body = read_dbc(path)
    cols = {}
    off = 1
    for name, ftype, flen in fields:
        col = [body[r * rsize + off: r * rsize + off + flen].decode("latin-1").strip() for r in range(nrec)]
        cols[name] = col
        off += flen
    return pd.DataFrame(cols)


def read_columns(path, names):
    """Fast column extraction with numpy (fixed-width bytes -> str arrays). Returns dict name -> np array of str."""
    import numpy as np
    fields, nrec, rsize, body = read_dbc(path)
    arr = np.frombuffer(body, dtype=np.uint8, count=nrec * rsize).reshape(nrec, rsize)
    off = 1
    out = {}
    for name, ftype, flen in fields:
        if name in names:
            col = np.ascontiguousarray(arr[:, off:off + flen])
            out[name] = np.char.strip(col.view(f"S{flen}").ravel().astype("U%d" % flen))
        off += flen
    out["_columns"] = [f[0] for f in fields]
    out["_n"] = nrec
    return out
