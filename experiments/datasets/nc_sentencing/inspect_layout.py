"""Inspect NC DAC OPI layouts and Sentence Component (read-only, untrusted data).
Run: python -I inspect_layout.py   (writes layouts.txt; prints a short report)
"""
import zipfile, pickle
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
CODES = ["OFNT3AA1", "APPT7AA1", "INMT4AA1", "INMT4BB1", "OFNT3BB1", "OFNT3CE1", "OFNT3DE1",
         "INMT4CA1", "INMT9CF1", "OFNT1BA1", "OFNT9BE1", "APPT9BJ1"]


def read_des(code):
    rows = []
    for line in (RAW / code / f"{code}.des").read_text(errors="replace").splitlines()[:]:
        if line.startswith("Name"):
            continue
        p = line.split()
        if len(p) < 5:
            continue
        name, typ, start, ln = p[0], p[-3], int(p[-2]), int(p[-1])
        rows.append((name, " ".join(p[1:-3]), typ, start, ln))
    return rows


def load(code, names):
    """Fixed-width read of selected fields, chunked; '?'-only values -> NaN."""
    lay = {r[0]: r for r in read_des(code)}
    specs = [(lay[n][3] - 1, lay[n][3] - 1 + lay[n][4]) for n in names]
    out = []
    for ch in pd.read_fwf(RAW / code / f"{code}.dat", colspecs=specs, names=names, dtype=str,
                          chunksize=500_000, header=None, na_filter=False):
        for c in ch.columns:
            ch[c] = ch[c].str.strip()
        out.append(ch)
    df = pd.concat(out, ignore_index=True)
    return df.replace(regex=r"^\?+$", value=pd.NA)


def dt(s):
    return pd.to_datetime(s.where(s != "0001-01-01"), errors="coerce")


def main():
    # 1. contents and layouts
    with open(HERE / "layouts.txt", "w") as f:
        for code in CODES:
            z = zipfile.ZipFile(RAW / f"{code}.zip")
            inside = [(i.filename, i.file_size) for i in z.infolist()]
            print(code, "zip contains:", "; ".join(f"{n} ({s/1e6:.1f} MB)" for n, s in inside))
            f.write(f"== {code}\n")
            for r in read_des(code):
                f.write("%-10s %-34s %-8s start=%-4d len=%d\n" % (r[0], r[1], r[2], r[3], r[4]))
    print("field lists written to layouts.txt")

    # 3. Sentence Component
    cache = RAW / "_sc_cache.pkl"
    cols = ["CMDORNUM", "CMPREFIX", "CMSENCMP", "PUNISHTYP", "CMSNTYPE", "PENCLASS", "PRIFELMD",
            "PRPOINTS", "PRLEVEL", "CMIMPSDT", "CMSENTDT", "CMOFFDTE", "CMCMPSTA", "CMDISPOS",
            "CMTRMMIN", "CMTRMMAX", "CMOFFNSE", "CMCOCNVT", "SSACTTRM"]
    if cache.exists():
        sc = pickle.load(open(cache, "rb"))
    else:
        sc = load("OFNT3CE1", cols)
        pickle.dump(sc, open(cache, "wb"))
    sc["conv"] = dt(sc.CMIMPSDT)
    n = len(sc)
    print(f"\nSC records: {n:,}; distinct offenders: {sc.CMDORNUM.nunique():,}")
    print("conviction date (CMIMPSDT) min/max:", sc.conv.min().date(), sc.conv.max().date(),
          "| n valid:", sc.conv.notna().sum())
    print("effective date (CMSENTDT) min/max:", dt(sc.CMSENTDT).min().date(), dt(sc.CMSENTDT).max().date())
    print("PRPOINTS: non-missing", sc.PRPOINTS.notna().sum(), "; values != '000':",
          (sc.PRPOINTS.dropna() != "000").sum())
    print("PRPOINTS top:", sc.PRPOINTS.value_counts(dropna=False).head(8).to_dict())
    print("PRLEVEL non-missing:", sc.PRLEVEL.notna().sum())
    sc.loc[sc.conv < "1970-01-01", "conv"] = pd.NaT
    print("conviction year counts (1980..2026, every 5th):",
          {int(y): int(v) for y, v in sc.conv.dt.year.value_counts().sort_index().items() if y % 5 == 0 and y >= 1980})
    for c, k in [("PUNISHTYP", 12), ("CMSNTYPE", 4), ("CMCMPSTA", 5)]:
        print(f"\n{c} (top {k}):")
        print(sc[c].value_counts(dropna=False).head(k).to_string())
    sc["pt"] = sc.PUNISHTYP.str.replace(r"\s+", " ", regex=True)
    sc["active"] = sc.pt.eq("ACTIVE SS")
    post = sc[sc.conv >= "2011-12-01"]  # JRA grid; conviction date used as proxy for offense date
    print(f"\nPost-2011 rows (conviction >= 2011-12-01): {len(post):,}; PUNISHTYP:")
    print(post.pt.value_counts(dropna=False).head(6).to_string())
    print("\nactive = PUNISHTYP 'ACTIVE SS'; prison per CMSNTYPE in post-2011 rows:",
          post.CMSNTYPE.value_counts().head(3).to_dict())
    fel = post[post.PENCLASS.str.fullmatch(r"CLASS [B-I]\d?( SS)?", na=False)]
    fel = fel[fel.PRLEVEL.notna() & fel.pt.isin(["ACTIVE SS", "INTERMEDIATE SS", "COMMUNITY SS (DCC)"])]
    print(f"Post-2011 felony SS rows with level and a SS punishment type: {len(fel):,}")
    fel = fel.assign(cls=fel.PENCLASS.str.replace("CLASS ", "").str.replace(" SS", ""),
                     lev=fel.PRLEVEL.str.replace("LEVEL ", ""))
    print("\nShare ACTIVE SS by felony class (rows) x prior record level (cols):")
    print(fel.pivot_table(index="cls", columns="lev", values="active", aggfunc="mean").round(2).to_string())
    print("Cell sizes:")
    print(fel.pivot_table(index="cls", columns="lev", values="active", aggfunc="size").to_string())
    fel = fel.assign(pts=pd.to_numeric(fel.PRPOINTS, errors="coerce"))
    band = pd.cut(fel.pts, [-1, 1, 5, 9, 13, 17, 10**6], labels=list("123456")).astype(str)
    lv = fel.lev.map({"I": "1", "II": "2", "III": "3", "IV": "4", "V": "5", "VI": "6"})
    print("\nShare of felony rows whose PRPOINTS falls in the band implied by PRLEVEL (I 0-1, II 2-5, III 6-9, IV 10-13, V 14-17, VI 18+):",
          round((band == lv).mean(), 3))
    print("Active share by PRPOINTS around the grid jump (class: level change at points; n in brackets):")
    for c, cut in [("E", 6), ("F", 10), ("G", 14), ("H", 18)]:
        d = fel[fel.cls == c]
        print(f" {c} (jump at {cut}):", "  ".join(
            f"{k}:{d[d.pts == k].active.mean():.2f}[{(d.pts == k).sum()}]" for k in range(cut - 3, cut + 3)))
    print("\nPost-2011 MISDEMEANOR SS rows by PRLEVEL (misdemeanor levels I-III use conviction counts):",
          post[post.PENCLASS.str.contains("MISDEMEANOR SS", na=False)].PRLEVEL.value_counts().to_dict())

    # commitments: new period of incarceration flag, admissions, repeat commitments
    cm = load("OFNT3BB1", ["CMDORNUM", "CMPREFIX", "CMADMDTE", "CMINTDTE", "CMNEWPD", "CMPRIOFF", "GMSEFFDT"])
    cm["adm"] = dt(cm.CMADMDTE)
    cm["intk"] = dt(cm.CMINTDTE)
    print(f"\nCommitments (OFNT3BB1): {len(cm):,}; CMNEWPD:", cm.CMNEWPD.value_counts(dropna=False).to_dict())
    print("valid CMADMDTE:", cm.adm.notna().sum(), " valid CMINTDTE:", cm.intk.notna().sum(),
          " adm range:", cm.adm.min().date(), cm.adm.max().date())
    print("commitments per offender (share with >=2):", round((cm.groupby("CMDORNUM").size() >= 2).mean(), 3))
    first = fel.groupby("CMDORNUM").conv.min().rename("first_conv")
    admin = cm[cm.adm.notna()].merge(first, left_on="CMDORNUM", right_index=True)
    later = admin[admin.adm > admin.first_conv + pd.Timedelta(days=30)].CMDORNUM.nunique()
    print(f"offenders with a post-2011 felony SS row: {len(first):,}; with a commitment admitted >30d after first such conviction: {later:,}")

    # 4. id matching
    ids = {}
    for code, col in [("INMT4AA1", "CIDORNUM"), ("APPT7AA1", "CDDORNUM"), ("OFNT3AA1", "CMDORNUM"),
                      ("OFNT3BB1", "CMDORNUM"), ("INMT4BB1", "CIDORNUM")]:
        ids[code] = set(load(code, [col])[col].dropna())
        print(f"{code} {col}: {len(ids[code]):,} distinct ids")
    sc_ids = set(sc.CMDORNUM.dropna())
    for code, s in ids.items():
        print(f"SC ids found in {code}: {len(sc_ids & s)/len(sc_ids):.3f}; "
              f"{code} ids found in SC: {len(sc_ids & s)/len(s):.3f}")
    both = ids["INMT4AA1"] & ids["APPT7AA1"]
    print(f"ids in both Inmate and P&P profiles: {len(both):,}")


main()
