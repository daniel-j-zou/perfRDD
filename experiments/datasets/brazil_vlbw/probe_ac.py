"""AC (Acre) 2018 linkage probe. Run: python -I probe_ac.py   (reads data/raw/dbc/*.dbc)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dbc
import pandas as pd
R = os.path.join(HERE, "data", "raw", "dbc")

dn18 = dbc.to_frame(os.path.join(R, "DNAC2018.dbc"))
print("SINASC AC 2018 births:", len(dn18), "has NUMERODN column:", "NUMERODN" in dn18.columns)
do = pd.concat([dbc.to_frame(os.path.join(R, f"DOAC{y}.dbc")).assign(file=y) for y in (2018, 2019)])
print("SIM AC deaths 2018+2019 (all ages, incl. fetal):", len(do), "has NUMERODN column:", "NUMERODN" in do.columns)
print("columns in SIM with 'DN' or 'NUM':", [c for c in do.columns if "NUM" in c or c.endswith("DN")])
print("columns in SINASC with 'DN' or 'NUM':", [c for c in dn18.columns if "NUM" in c or c.endswith("DN")])
print("TIPOBITO counts:", do.TIPOBITO.value_counts().to_dict())
print("IDADE first digit (non-fetal):", do[do.TIPOBITO == "2"].IDADE.str[:1].value_counts().to_dict())

# infant deaths: non-fetal, unit 0 (min),1 (hours),2 (days),3 (months) -> age < 1 year (DATASUS coding: 0 min,1 h,2 d,3 m,4 y)
nf = do[do.TIPOBITO == "2"].copy()
inf = nf[nf.IDADE.str[:1].isin(list("0123"))].copy()
inf["by"] = inf.DTNASC.str[4:8]
print("infant deaths by file year:", inf.file.value_counts().to_dict())
print("infant deaths by birth year:", inf.by.value_counts().sort_index().to_dict())
c18 = inf[inf.by == "2018"].copy()
print("infant deaths, born 2018, died 2018 or 2019:", len(c18))

def miss(s, codes=()):
    s = s.fillna("")
    return int((s.str.strip() == "").sum()), int(s.isin(codes).sum())

print("\n--- SINASC AC 2018 availability (blank, coded-ignored) ---")
spec = {"PESO": ("9999", "0000"), "GESTACAO": ("9",), "SEMAGESTAC": ("99",), "GRAVIDEZ": ("9",), "SEXO": ("0", "9"),
        "IDADEMAE": ("99",), "ESCMAE": ("9",), "ESCMAE2010": ("9",), "QTDFILVIVO": ("99",), "QTDFILMORT": ("99",),
        "QTDGESTANT": ("99",), "PARIDADE": ("9",), "CONSULTAS": ("9",), "CONSPRENAT": ("99",), "MESPRENAT": ("99",),
        "APGAR1": ("99",), "APGAR5": ("99",), "RACACOR": ("9",), "PARTO": ("9",), "ESTCIVMAE": ("9",), "IDADEPAI": ("99",)}
n = len(dn18)
for k, v in spec.items():
    b, c = miss(dn18[k], v)
    print(f"{k:12s} blank {b:6d} ({100*b/n:5.1f}%)  coded-ignored {c:6d} ({100*c/n:5.1f}%)  usable {n-b-c:6d} ({100*(n-b-c)/n:5.1f}%)")
w = pd.to_numeric(dn18.PESO, errors="coerce")
print("PESO <1500:", int((w < 1500).sum()), f"({100*(w<1500).mean():.2f}% of births); <2500:", int((w < 2500).sum()), "; PESO 500-1499:", int(((w>=500)&(w<1500)).sum()))
print("PESO quantiles:", w.quantile([0, .01, .5, .99, 1]).to_dict())
print("heaping: share of PESO divisible by 100:", round(float((w.dropna() % 100 == 0).mean()), 3), " by 50:", round(float((w.dropna() % 50 == 0).mean()), 3))
print("GESTACAO:", dn18.GESTACAO.value_counts().sort_index().to_dict())
print("RACACOR:", dn18.RACACOR.value_counts().sort_index().to_dict())
print("CODMUNRES in AC (prefix 12):", int(dn18.CODMUNRES.str[:2].eq("12").sum()))

print("\n--- infant-death block fields in SIM (cohort born 2018, n=%d) ---" % len(c18))
for k, v in {"DTNASC": (), "SEXO": ("0", "9"), "PESO": ("9999", "0000"), "IDADEMAE": ("99",), "CODMUNRES": (), "SEMAGESTAC": ("99",),
             "GESTACAO": ("9",), "GRAVIDEZ": ("9",), "QTDFILVIVO": ("99",), "ESCMAE": ("9",), "RACACOR": ("9",)}.items():
    b, c = miss(c18[k], v)
    print(f"{k:12s} blank {b:4d}  coded-ignored {c:4d}  usable {len(c18)-b-c:4d} ({100*(len(c18)-b-c)/len(c18):5.1f}%)")

# probabilistic keys
def num(s):
    return pd.to_numeric(s, errors="coerce")
dn = dn18.copy()
for d in (dn, c18):
    d["peso"] = num(d.PESO)
    d["ima"] = num(d.IDADEMAE)
dn["peso"] = dn.peso.where(dn.peso > 0)
c18["peso"] = c18.peso.where(c18.peso > 0)
ok_sex = lambda d: d.SEXO.isin(["1", "2"])
keysets = {
    "DTNASC+SEXO": ["DTNASC", "SEXO"],
    "DTNASC+SEXO+PESO": ["DTNASC", "SEXO", "peso"],
    "DTNASC+SEXO+IDADEMAE": ["DTNASC", "SEXO", "ima"],
    "DTNASC+SEXO+PESO+IDADEMAE": ["DTNASC", "SEXO", "peso", "ima"],
    "DTNASC+SEXO+PESO+IDADEMAE+CODMUNRES": ["DTNASC", "SEXO", "peso", "ima", "CODMUNRES"],
    "DTNASC+SEXO+PESO+CODMUNRES": ["DTNASC", "SEXO", "peso", "CODMUNRES"],
}
print("\n--- exact-key match of the %d infant deaths against %d SINASC AC-2018 births ---" % (len(c18), len(dn)))
print("(CODMUNRES: SINASC 7 digits vs SIM 6 digits is trimmed to 6)")
dn["CODMUNRES"] = dn.CODMUNRES.str[:6]
c18["CODMUNRES"] = c18.CODMUNRES.str[:6]
rows = []
for name, ks in keysets.items():
    d = c18.dropna(subset=[k for k in ks if k in ("peso", "ima")])
    d = d[(d[[k for k in ks if k not in ("peso", "ima")]] != "").all(axis=1)]
    if "SEXO" in ks:
        d = d[ok_sex(d)]
    s = dn.dropna(subset=[k for k in ks if k in ("peso", "ima")])
    cnt = s.groupby(ks).size().rename("n_sinasc").reset_index()
    m = d.merge(cnt, on=ks, how="left")
    m["n_sinasc"] = m.n_sinasc.fillna(0).astype(int)
    # also: share of SINASC births whose key is unique
    uniq_share = float((s.groupby(ks).size() == 1).sum() / len(s))
    rows.append((name, len(d), int((m.n_sinasc == 0).sum()), int((m.n_sinasc == 1).sum()), int((m.n_sinasc > 1).sum()), uniq_share))
print(f"{'key':40s} {'deaths w/ key':>13s} {'0 match':>8s} {'1 match':>8s} {'>1':>6s}  share SINASC keys unique")
for r in rows:
    print(f"{r[0]:40s} {r[1]:13d} {r[2]:8d} {r[3]:8d} {r[4]:6d}  {r[5]:.3f}")
# among deaths with 1 match: weight agreement check is trivially true; report VLBW deaths
v = c18[c18.peso < 1500]
print("\ninfant deaths with PESO<1500 in SIM (cohort 2018):", len(v), "; SINASC births <1500:", int((dn.peso < 1500).sum()))
