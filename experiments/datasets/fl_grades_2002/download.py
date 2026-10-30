"""Download FLDOE's school-grades spreadsheets and technical assistance papers, 2001-2014.

Every file comes from FLDOE's public file CDN, folder ``cdn.fldoe.org/file/18534/``: the
folder behind https://www.fldoe.org/accountability/accountability-reporting/school-grades/
and its archive page. No login, form or CAPTCHA, and robots.txt on cdn.fldoe.org allows
everything. The archive page itself (www.fldoe.org/.../archives.stml) returns 403 to
scripts, so it was not used: the file names were found from web-search results and from
the Internet Archive's index of the folder (web.archive.org CDX query for
``fldoe.org/core/fileparse.php/18534/urlt/*``), and every file below then downloads from
the CDN with a plain ``urllib`` request (Python's default User-Agent; no header is set).

Files, by spring year of the school year (2002 = 2001-02):

* ``SchoolGrades{yy}.xls``: one row per graded school, 2001-2009. From 2010 FLDOE split
  the release into ``NonHighSchoolGrades{yy}.xls`` (elementary, middle and combination
  schools without grade 12, still on the FCAT-only 800-point model) and
  ``HighSchoolGrades{yy}.xls`` (schools with grade 12, on the 1600-point model with
  graduation rate and acceleration). ``SchoolGrades10.xls`` and ``SchoolGrades11.xls``
  do not exist (404, checked 2026-10-29).
* ``SchoolGradesCalcTAP{yy}.pdf``: the technical assistance paper ("Guide to
  Calculating School Grades") for 2003-2013; ``SchoolGradesCalcGuide14.pdf`` for 2014.
  ``SchoolGradesCalcTAP02.pdf`` does not exist (404).
* ``{yy}{yy}GuideSheet.pdf`` (2001-02 to 2008-09) and ``{yy}{yy}SchoolGradesGuideSheet.pdf``
  (2009-10 to 2013-14): FLDOE's one-page summaries of each year's rule; the 2001-02 sheet
  is the only official statement of the 2002 rule found.

Idempotent: a file is skipped when it is already present and non-empty. A download is
written to ``<name>.part`` first and then renamed, so an interrupted run leaves no
truncated file behind.

    python -m experiments.datasets.fl_grades_2002.download          # everything
    python -m experiments.datasets.fl_grades_2002.download --check  # sizes and SHA-256

The NCES CCD files (2001-02 to 2013-14 public school universe, about 150 MB) supply the
covariates; Florida rows are read from the zips in place.
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
CDN = "https://cdn.fldoe.org/file/18534"

# Spring year -> spreadsheet file names (checked 2026-10-29).
SPREADSHEETS = {y: [f"SchoolGrades{y % 100:02d}.xls"] for y in range(2001, 2010)}
SPREADSHEETS.update({y: [f"NonHighSchoolGrades{y % 100:02d}.xls",
                         f"HighSchoolGrades{y % 100:02d}.xls"] for y in range(2010, 2015)})
# Combined files that also exist for 2012-2014; downloaded for cross-checks only.
COMBINED = {y: f"SchoolGrades{y % 100:02d}.xls" for y in (2012, 2013, 2014)}
TAPS = {y: f"SchoolGradesCalcTAP{y % 100:02d}.pdf" for y in range(2003, 2014)}
TAPS[2014] = "SchoolGradesCalcGuide14.pdf"
GUIDE_SHEETS = {y: f"{(y - 1) % 100:02d}{y % 100:02d}GuideSheet.pdf" for y in range(2002, 2010)}
GUIDE_SHEETS.update({y: f"{(y - 1) % 100:02d}{y % 100:02d}SchoolGradesGuideSheet.pdf"
                     for y in range(2010, 2015)})


# NCES Common Core of Data (CCD), Public Elementary/Secondary School Universe Survey, by
# spring year: (data zip, record layout). 2001-02 to 2006-07 are fixed-width files split
# by state range (``ai`` = Alabama to Iowa, which holds Florida); 2007-08 onward are
# tab-delimited national files with a header row. Listed on
# https://nces.ed.gov/ccd/pubschuniv.asp (checked 2026-10-29); the latest version of each
# year is used (e.g. 2009-10 version 2a).
NCES = "https://nces.ed.gov/ccd"
CCD = {
    2002: ("data/zip/sc011aai_dat.zip", "data/txt/psu01lay.txt"),
    2003: ("data/zip/sc021aai_dat.zip", "data/txt/psu021alay.txt"),
    2004: ("data/zip/sc031aai_dat.zip", "data/txt/psu031alay.txt"),
    2005: ("data/zip/sc041bai_dat.zip", "data/txt/psu041blay.txt"),
    2006: ("data/zip/sc051aai_dat.zip", "data/txt/psu051alay.txt"),
    2007: ("data/zip/sc061cai_dat.zip", "data/txt/psu061clay.txt"),
    2008: ("data/zip/sc071b_txt.zip", "data/txt/psu071blay.txt"),
    2009: ("data/zip/sc081b_txt.zip", "data/txt/psu081blay.txt"),
    2010: ("data/zip/sc092a_txt.zip", "data/txt/psu092alay.txt"),
    2011: ("Data/zip/sc102a_txt.zip", "Data/txt/psu102alay.txt"),
    2012: ("Data/zip/sc111a_supp_txt.zip", "Data/txt/sc111alay.txt"),
    2013: ("Data/zip/sc122a_txt.zip", "Data/txt/sc122alay.txt"),
    2014: ("Data/zip/sc132a_txt.zip", "Data/txt/sc132alay.txt"),
}


def ccd_paths(year: int) -> tuple[Path, Path]:
    data, layout = CCD[year]
    return RAW / "ccd" / Path(data).name, RAW / "ccd" / Path(layout).name


def jobs() -> list[tuple[str, Path]]:
    out = [(f"{CDN}/{n}", RAW / n) for names in SPREADSHEETS.values() for n in names]
    out += [(f"{CDN}/{n}", RAW / n) for n in COMBINED.values()]
    out += [(f"{CDN}/{n}", RAW / "guides" / n) for n in TAPS.values()]
    out += [(f"{CDN}/{n}", RAW / "guides" / n) for n in GUIDE_SHEETS.values()]
    for y, (data, layout) in CCD.items():
        dpath, lpath = ccd_paths(y)
        out += [(f"{NCES}/{data}", dpath), (f"{NCES}/{layout}", lpath)]
    return out


# (bytes, SHA-256) of every file as downloaded on 2026-10-29, keyed by its path under
# data/raw/. ``--check`` compares the files on disk with this list.
MANIFEST = {
    "SchoolGrades01.xls": (626176, "092d3bb5a08681e491b698623b5206b91c74e310a26e9b3e7e36f2ef3a978a5a"),
    "SchoolGrades02.xls": (531456, "1644e7910c6f838d2cb563092f4c01fc6d37135692f98a2517460e825caf3700"),
    "SchoolGrades03.xls": (871936, "bf48a8eb243e2f05c27f7e745050490165258c854c25e771893a741d7130c273"),
    "SchoolGrades04.xls": (888832, "17c14df77ec05889657367cf52a2c5d4e77ede7ad19643346a99a8c00e0b84b7"),
    "SchoolGrades05.xls": (1089024, "627782ba60c99f9de4ff0c5032953d7e7d4003922493c221abf582d29ea32f2b"),
    "SchoolGrades06.xls": (1024000, "bf161b454d28726ad458a553e541b2050bcb64865b0f2153930b37e186f4c91a"),
    "SchoolGrades07.xls": (1492992, "2eeb870cf1f8c1133991122e798646653a79ead4d1cd1cba8103660bb6528d75"),
    "SchoolGrades08.xls": (1395200, "755b644e90c809ae7f3c9c20772434a3facc2b55445ace5417b07901641e1b8c"),
    "SchoolGrades09.xls": (1479680, "5fb404a10338a2d450cd50b1235b69689b975a12dac082bd33d29af5c8f88b75"),
    "NonHighSchoolGrades10.xls": (1448960, "d60ea6c1d44d100c512494fa4fdc2b134f631b36a27a8993b74ff0c485cc33a7"),
    "HighSchoolGrades10.xls": (380416, "d96dfe44f7ccc30b11b1959b6de9c1cfddcba9a2662a8944b3df09599760c4cf"),
    "NonHighSchoolGrades11.xls": (1398784, "096cc82704376d7e5faeaf49f9c664428a5851c0191f8e83778825be703cf0c2"),
    "HighSchoolGrades11.xls": (770560, "a06572811706b45b0f9098a25fa09e3f28589c22ccac000fd440fc49a57a00af"),
    "NonHighSchoolGrades12.xls": (1668608, "0f9a89c7d02b7298ab964ff70ff40dc520524c42bc2ce98dfaca73a10feb1954"),
    "HighSchoolGrades12.xls": (544256, "d86bf8d9c0edcce65ee16ffb57930a4499d5cc20fdf39fbb051d7e6fb737b05c"),
    "NonHighSchoolGrades13.xls": (2123776, "56e30d9ba7699607d2dc1663010252a8d57372c85745dc11666637db845d607c"),
    "HighSchoolGrades13.xls": (552960, "811dd6415835a2d1e4805368fd7d184c3009e681ccf6cee8772081add9632dae"),
    "NonHighSchoolGrades14.xls": (2143232, "ddd6e8673e1d0fc03fa6dc8284daefa60d23e2928409401935b3d84f827c28df"),
    "HighSchoolGrades14.xls": (647168, "ba900599c95be576ea98011bcea98c32e6df470829dabb478f47dc98442372d0"),
    "SchoolGrades12.xls": (1271296, "cf1874c8293bd2c883d0d770b87256906ad2370fd788942ffe37527cd7757d2b"),
    "SchoolGrades13.xls": (1404928, "71eaafea0c5ea631a712d2bbde99ae30bdb4ef98e41c82079a7b12263874091a"),
    "SchoolGrades14.xls": (1497600, "de45e8d98d8c86317ea2f0cc36d1b6546db96aa81db0e9a2eaf0ad3cedb7df08"),
    "guides/SchoolGradesCalcTAP03.pdf": (767077, "16a1c5dd20a075cf3f60c6ef10f79bb8e2da569e445c1a2e831873008c4254ac"),
    "guides/SchoolGradesCalcTAP04.pdf": (190932, "dbb5a61dd3bab13b0620a028c5ba3b230b77be9e6dff846382553fd1dee68cc4"),
    "guides/SchoolGradesCalcTAP05.pdf": (802164, "5268bf07397fb829d9c4ffd3f622eb2c9030a9d3e66e6e6b13cf38d099d79a47"),
    "guides/SchoolGradesCalcTAP06.pdf": (180750, "6856e2e0ab17b163d0b01d73f6e5d78d319b51c8b2e65a9354826c4dac3e5ed1"),
    "guides/SchoolGradesCalcTAP07.pdf": (472879, "37bc6fe7b92f87b748693532cd81b97ef9e3d24616a31491a219296fa593d62c"),
    "guides/SchoolGradesCalcTAP08.pdf": (601708, "8638993259761f366c9a847b7e61d9ce531a9e5314b3aa307fd3cd6926467177"),
    "guides/SchoolGradesCalcTAP09.pdf": (445299, "b7fcf5f8ca1b1804e2df892fb11de535e768832a421585c8b3d68f7d12e1cbcd"),
    "guides/SchoolGradesCalcTAP10.pdf": (414556, "444858d48282597e2956f94305b0f067da716b2122feabb47389fb666be3c969"),
    "guides/SchoolGradesCalcTAP11.pdf": (311274, "0f2410560ef150e498bf453fdb9d5f481bd2e0f4869ddd17ae155befe6996714"),
    "guides/SchoolGradesCalcTAP12.pdf": (771067, "3db5534c9851c603d24b89daa24e927e70983742c27438dc55d9bdb98d382633"),
    "guides/SchoolGradesCalcTAP13.pdf": (1245604, "7b749ccb88331d0230ef08f30d158ba2271b71ebf2606dfd219947f73c9d6139"),
    "guides/SchoolGradesCalcGuide14.pdf": (1247453, "50a2cb447def1040257ee152a5c6c23681f70f147fc2744b5b5b66d692010ba2"),
    "guides/0102GuideSheet.pdf": (186839, "cc63b3d64167a499a6287136b739f4cc44a744e9e7f895bf5685b22d7a348954"),
    "guides/0203GuideSheet.pdf": (227693, "6b8f9b1bea5fcaf9fd12f7549053c9173934485112dbcf010d9187efe5af3612"),
    "guides/0304GuideSheet.pdf": (107676, "1ef6f4b07f33ca5e93f90d3f97dce6ce96b8be1a331f3dafa56d4ac9214a8a4c"),
    "guides/0405GuideSheet.pdf": (215151, "aecba01eb0e08bb4ae2fa4816a835ca2eef43ea051dff448bfe087c6e4c1fd5b"),
    "guides/0506GuideSheet.pdf": (221307, "3ab76fb0005448f5618ddafe024873401e135b1a31fc96b0739d80ac66c2666b"),
    "guides/0607GuideSheet.pdf": (1491389, "b5b385d8fced6701dc78a28e33209e077863203d885a2359db2c0b9fab5b2495"),
    "guides/0708GuideSheet.pdf": (453768, "b4fa80273c6ecca009da1e6251e84ccb5e0f463297543b86d4769294076e5086"),
    "guides/0809GuideSheet.pdf": (521123, "a9efb6ec6ac54a4cebbbff3a7e5ac8cd112616759fdf0b66b60d71b15fa4062a"),
    "guides/0910SchoolGradesGuideSheet.pdf": (81690, "9ea201f9c06357cfd37a42067335ef8232a15366396ae4883c80df9bfdbc648f"),
    "guides/1011SchoolGradesGuideSheet.pdf": (85766, "1b272c87d7847c6c2df414b977b0b821edf0c7f018dec0b88c91b5e929fcb762"),
    "guides/1112SchoolGradesGuideSheet.pdf": (96233, "61177caede66e451e63d27b721725d54eda35d2eccfd4faecca6deac9dcfb4e8"),
    "guides/1213SchoolGradesGuideSheet.pdf": (261377, "f120113d52b611aca413eb2395f905affdf58c5d905e5e2aa9210a77fe0b240c"),
    "guides/1314SchoolGradesGuideSheet.pdf": (291522, "eba6ebd494976f76248477d290ca127c9c7621ec1a1d754db941b1d3151f5cca"),
    "ccd/sc011aai_dat.zip": (5823835, "d48b4eef96fae41b3f0531e0429c640adf1fe3b2ce321993d782ff852b0ad1ab"),
    "ccd/psu01lay.txt": (103548, "b76c1921e01649dc7e96eb88b338297436812dfd74f1f044ef62c8885db8d10d"),
    "ccd/sc021aai_dat.zip": (6132461, "0c17dafcd1bff741c652acf01e49ccc5dc814240e7ee5408b1f20e47c0bdce13"),
    "ccd/psu021alay.txt": (103340, "e280fb64648b81dce0976c00a0539e90c51904755f84e7208b4e80cfb4a318f0"),
    "ccd/sc031aai_dat.zip": (7013237, "d3f40a74c962fa54b7fd545eb13e3e913588c7dcef74c93fd7fe59b44a28f78b"),
    "ccd/psu031alay.txt": (103340, "c7c6f66a18de6d6d1dad02c3efe5550fd1b313fea65e9ae73383af6a59310cf7"),
    "ccd/sc041bai_dat.zip": (6619713, "466146a306cafa0708b909540bd3f03f32e29c9ff8ce23613d0df99301e81360"),
    "ccd/psu041blay.txt": (145170, "6aa3cb7fef0b43ee6fd96735df1b0419ff3f9481a9b41fb9938ff8cafd8d09a0"),
    "ccd/sc051aai_dat.zip": (5766231, "33cf1f2c6440444f44130e85b87afb9087980d1f539f38db200500e166d05d9b"),
    "ccd/psu051alay.txt": (103465, "fe6433ac7515c6e15d0c3414f3a11b54b6fafca543e00ddc63c81eed9bb5442a"),
    "ccd/sc061cai_dat.zip": (5794507, "b288fb241a26ec10f1d577d45bbac17b10a5beaa456c7e2e7a1e707d5ef8b77d"),
    "ccd/psu061clay.txt": (116107, "e376a5bc94491a1036e28ff320e0572fca4bcc3bd6759e08576ec3c4e3f2bd2f"),
    "ccd/sc071b_txt.zip": (14265030, "357a41bc66cfb12676bd265705fc4d09b7b5b52acaa94597be1b469df007fd53"),
    "ccd/psu071blay.txt": (26932, "e8fea26f34bd23b290e0a373f647c4810c942ebfdd073b68fe532ebb61d8c83a"),
    "ccd/sc081b_txt.zip": (14857249, "a500dc7d255bc82db7ea2572bdb3628a03514836f2d253432763a0889f41a629"),
    "ccd/psu081blay.txt": (39426, "0b614060a067d774424ae8f96a349f1a7eec6f3f9d76357ef9688f560a3bbda7"),
    "ccd/sc092a_txt.zip": (15748340, "7672e38cf821f638d8f10e0b6f9b81d97a52b485761d00022cff65b3ba1619a3"),
    "ccd/psu092alay.txt": (39586, "c713cafdd52f6e104627e17a3b742d5fed4c029daf405dbd5cc8f2f26efa7aa4"),
    "ccd/sc102a_txt.zip": (15582802, "aed6bfd8d717993a11cb43d3099a678e0aaef0ac6f0b7374a3f0e53fdf6f84c3"),
    "ccd/psu102alay.txt": (41557, "e3272bea003758fd4bb8a0f245e6515c76ddf96902eab8f415ea55c117d794b0"),
    "ccd/sc111a_supp_txt.zip": (15844597, "57967ac5882ab9559cb9ec58458a31aa218988319465751798fccf014732c6b5"),
    "ccd/sc111alay.txt": (42422, "27be1dee50f348eca0e01d2a55f2e569f5c385880b259bbc89f029c530e52447"),
    "ccd/sc122a_txt.zip": (14772012, "c68345e94b886bc585dfe3487dc44efefd3d01222770739c62827b228bb46dba"),
    "ccd/sc122alay.txt": (41992, "f4dbc93998d8e7e10759f6e7e5a19a9a23caf92062e403184119204e1e59fcfa"),
    "ccd/sc132a_txt.zip": (14732901, "a0783a59d470d66572ae9a8b3559ccabc9279c6eade91aafe9b2422fce7d5fa9"),
    "ccd/sc132alay.txt": (42178, "ae818b55df0238f5d2fb2dba761165e04c417f10c867fd50ad4b930e6cbe5a86"),
}


def fetch(url: str, dest: Path) -> int:
    if dest.exists() and dest.stat().st_size > 0:
        return dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(part, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    os.replace(part, dest)
    return dest.stat().st_size


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str]) -> None:
    check = "--check" in argv
    total, changed = 0, 0
    for url, dest in jobs():
        size = (dest.stat().st_size if dest.exists() else 0) if check else fetch(url, dest)
        digest = sha256(dest) if dest.exists() else "-"
        ok = MANIFEST.get(str(dest.relative_to(RAW))) == (size, digest)
        changed += not ok
        total += size
        print(f"{size:>11,d}  {digest[:16]}  {'ok ' if ok else 'NEW'}  {url}")
    print(f"{total:>11,d}  total bytes; {changed} file(s) differ from MANIFEST")


if __name__ == "__main__":
    main(sys.argv[1:])
