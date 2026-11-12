"""National sizes from the .dbc files in data/raw/dbc. Run: python -I national.py"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dbc
R = os.path.join(HERE, "data", "raw", "dbc")

for y in ((17, 18, 19) if "--doinf" in sys.argv else ()):
    f = os.path.join(R, f"DOINF{y}.dbc")
    c = dbc.read_columns(f, {"DTNASC", "IDADE", "PESO", "IDADEMAE", "SEXO", "TIPOBITO", "CODMUNRES", "SEMAGESTAC"})
    n = c["_n"]
    print(f"DOINF{y}: records {n}; NUMERODN in columns: {'NUMERODN' in c['_columns']}; columns: {len(c['_columns'])}")
    idade = c["IDADE"]; d1 = np.char.ljust(idade, 1).astype("U1")
    print("  IDADE unit digit:", dict(zip(*np.unique(d1, return_counts=True))))
    print("  TIPOBITO:", dict(zip(*np.unique(c["TIPOBITO"], return_counts=True))) if "TIPOBITO" in c else "n/a")
    by = np.char.ljust(c["DTNASC"], 8).astype("U8"); by = np.array([s[4:8] for s in by])
    u, k = np.unique(by, return_counts=True); print("  birth year counts (top):", dict(zip(u[-4:], k[-4:])), "blank:", int((c["DTNASC"] == "").sum()))
    p = np.array([int(s) if s.isdigit() else -1 for s in c["PESO"]])
    print("  PESO present(>0):", int((p > 0).sum()), f"({100*(p>0).mean():.1f}%); <1500:", int(((p > 0) & (p < 1500)).sum()))
    print("  IDADEMAE present:", int(((c['IDADEMAE'] != '') & (c['IDADEMAE'] != '99')).sum()), " SEXO 1/2:", int(np.isin(c['SEXO'], ['1', '2']).sum()))

f = os.path.join(R, "DNBR2017.dbc")
c = dbc.read_columns(f, {"PESO", "DTNASC", "SEXO", "IDADEMAE", "GESTACAO", "SEMAGESTAC", "CODMUNRES"})
n = c["_n"]
p = np.array([int(s) if s.isdigit() else -1 for s in c["PESO"]])
by = np.array([s[4:8] for s in np.char.ljust(c["DTNASC"], 8).astype("U8")])
print(f"DNBR2017: births {n}; NUMERODN in columns: {'NUMERODN' in c['_columns']}")
print("  birth years:", dict(zip(*np.unique(by, return_counts=True))))
print("  PESO valid(>0):", int((p > 0).sum()), "; <1500:", int(((p > 0) & (p < 1500)).sum()), "; 500-1499:", int(((p >= 500) & (p < 1500)).sum()), "; <2500:", int(((p > 0) & (p < 2500)).sum()))
print("  PESO exactly 1500:", int((p == 1500).sum()), "; 1490-1499:", int(((p >= 1490) & (p < 1500)).sum()), "; 1500-1509:", int(((p >= 1500) & (p < 1510)).sum()), "; 1400:", int((p==1400).sum()), "; 1600:", int((p==1600).sum()))
print("  GESTACAO usable:", int(np.isin(c["GESTACAO"], list("123456")).sum()), "; SEMAGESTAC usable:", int(((c["SEMAGESTAC"] != "") & (c["SEMAGESTAC"] != "99")).sum()))
