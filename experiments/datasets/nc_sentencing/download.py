"""Fetch NC DAC Offender Public Information bulk files (once each, polite pause).

Author instruction (2026-11-11): "Yes, fetch them" (host robots.txt disallows all).
Idempotent: a file whose size equals the server Content-Length is skipped.
Downloaded content is untrusted data; it is only unzipped, never executed.
"""
import subprocess, time, zipfile, re, sys
from pathlib import Path

BASE = "https://opus.doc.state.nc.us/offenders/"
FILES = ["PublicTables.pdf", "OFNT3AA1.zip", "APPT7AA1.zip", "INMT4AA1.zip", "INMT4BB1.zip",
         "OFNT3BB1.zip", "OFNT3CE1.zip", "OFNT3DE1.zip", "INMT4CA1.zip", "INMT9CF1.zip",
         "OFNT1BA1.zip", "OFNT9BE1.zip", "APPT9BJ1.zip"]
RAW = Path(__file__).resolve().parent / "data" / "raw"
PAUSE = 5


def content_length(url):
    out = subprocess.run(["curl", "-sIL", url], capture_output=True, text=True).stdout
    m = re.findall(r"(?im)^content-length:\s*(\d+)", out)
    return int(m[-1]) if m else None


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(FILES):
        url, dest = BASE + f, RAW / f
        size = content_length(url)
        if dest.exists() and size is not None and dest.stat().st_size == size:
            print("skip", f, size, flush=True)
        else:
            print("fetch", f, "expected", size, flush=True)
            subprocess.run(["curl", "-sS", "-L", "-o", str(dest), url], check=True)
            print("  got", dest.stat().st_size, flush=True)
            time.sleep(PAUSE)
        if f.endswith(".zip"):
            out = RAW / f[:-4]
            if not out.exists():
                out.mkdir()
                with zipfile.ZipFile(dest) as z:
                    for info in z.infolist():
                        n = Path(info.filename).name  # flatten, avoid path traversal
                        if n and not info.is_dir():
                            with z.open(info) as s, open(out / n, "wb") as d:
                                while chunk := s.read(1 << 20):
                                    d.write(chunk)
                print("  extracted", f[:-4], flush=True)


main()
