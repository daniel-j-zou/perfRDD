"""Download the CMS Nursing Home Compare monthly snapshots (NBER mirror), CMS's state
health-inspection cut points and the Five-Star Technical Users' Guides.

Every file is public, with no login, form or CAPTCHA, and answered a plain urllib request
with an honest user agent on 2026-10-28:

* ``snapshots``: the monthly Nursing Home Compare "download the database" archives that
  NBER mirrors at https://data.nber.org/compare/nursing-home/. Folders 2013/08-2015/06
  hold ``DMG_CSV_DOWNLOAD<YYYYMM>01.zip``; folders 2015/07-2019/11 hold the same archive
  saved under the name ``application%2Fzip%3B%20charset%3Dbinary`` (a gzip-wrapped zip).
  Each archive is 18.9-34.3 MB (1.85 GB in all). Only these members are kept, gzip-compressed, in
  ``data/raw/snapshots/<YYYY_MM>/`` (YYYY_MM = the NBER folder):
    - ``ProviderInfo_Download.csv``: one row per nursing home, with the star ratings, the
      three survey cycles' deficiency scores and the weighted health-inspection score, the
      reported / case-mix / adjusted staffing hours and the facility characteristics;
    - ``StateAverages_Download.csv`` (``StateUSAverages_Download.csv`` from 2016), and
    - ``Ownership_Download.csv`` for the first snapshot of each calendar year (chain).
  The archive is deleted after extraction (``--keep-archives`` keeps it). Its URL, size
  and SHA-256 are recorded below and in ``data/raw/snapshots/manifest.csv``.
  The folders ``YYYY/MM/download/`` on the mirror hold stale June-2013 "flat files"
  (NHCRatings_flatfiles.zip and the like, the same bytes in every month) and are not used.
  The mirror has no snapshots before 2013/08 except QM files, and none after 2019/11.
* ``cutpoints``: CMS's state-level health-inspection cut points, monthly, from
  www.cms.gov (``Health-Inspection-Cut-Points-2008-2016.zip``: SAS files and RTF tables,
  January 2008 - October 2016) and the monthly "State-Level Cut Point Tables" PDF
  (``cutpointstable.pdf``), which CMS overwrites every month, as captured by the Internet
  Archive (web.archive.org, ``id_`` raw captures): one capture per table month,
  September 2016 to November 2019.
* ``guides``: the Five-Star Quality Rating System Technical Users' Guide
  (``usersguide.pdf``), which CMS also overwrites in place, as captured by the Internet
  Archive: the versions of July 2012, February 2015 (two captures), July 2016, January
  2017, February 2018, April 2018, July 2018, April 2019 (two captures), October 2019 and
  April 2020.
* ``crosswalk``: NBER's SSA-to-FIPS county crosswalk for FY2016 with the CMS CBSA
  (https://data.nber.org/ssa-fips-state-county-crosswalk/2016/), for an urban indicator.

data.cms.gov (the Provider Data Catalog and its archived monthly zips) answers "Access
Denied" to scripts (checked 2026-10-27), so nothing here comes from it.

Idempotent: a file (or a snapshot's extracted ProviderInfo) is skipped when it is already
present and non-empty. Downloads are written to ``<name>.part`` in the same folder and
renamed, so an interrupted run leaves no truncated file behind. The SHA-256 of each file
on 2026-10-28 is recorded; a different hash is reported, not raised.

    python -m experiments.datasets.nursing_homes.download                     # everything
    python -m experiments.datasets.nursing_homes.download --only guides cutpoints
    python -m experiments.datasets.nursing_homes.download --check             # sizes, hashes
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import os
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
SNAP = RAW / "snapshots"
UA = {"User-Agent": "PerfRDD-research-download/1.0 (python urllib)"}
DOWNLOADED = "2026-10-28"

NBER = "https://data.nber.org/compare/nursing-home"
APPZIP = "application%252Fzip%253B%2520charset%253Dbinary"
CMS_DL = ("https://www.cms.gov/Medicare/Provider-Enrollment-and-Certification/"
          "CertificationandComplianc/Downloads")
WAYBACK = "https://web.archive.org/web"

# NBER folder (YYYY_MM) -> archive URL. Folders that repeat an earlier archive (2013/10,
# 2013/12, 2014/10) and the "20150301TESTING" archive are left out; 2014/05 has none.
_DMG = ["2013_08", "2013_09", "2013_11", "2014_01", "2014_02", "2014_03", "2014_04",
        "2014_06", "2014_07", "2014_08", "2014_09", "2014_11", "2014_12", "2015_01",
        "2015_02", "2015_03", "2015_04", "2015_05", "2015_06"]
_APP = ([f"2015_{m:02d}" for m in range(7, 13)]
        + [f"{y}_{m:02d}" for y in (2016, 2017, 2018) for m in range(1, 13)]
        + [f"2019_{m:02d}" for m in range(1, 12)])


def _folder(tag: str) -> str:
    y, m = tag.split("_")
    # The mirror uses zero-padded months up to 2016/09 and plain months from 2017 (2016 has
    # both); 2015/07-2016/12 resolve with zero padding.
    return f"{y}/{m}" if int(y) <= 2016 else f"{y}/{int(m)}"


SNAPSHOTS: dict[str, str] = {
    **{t: f"{NBER}/{_folder(t)}/DMG_CSV_DOWNLOAD{t.replace('_', '')}01.zip" for t in _DMG},
    **{t: f"{NBER}/{_folder(t)}/{APPZIP}" for t in _APP},
}
# The first snapshot of each calendar year also keeps the ownership file (chain indicator).
OWNERSHIP_TAGS = ("2013_08", "2014_01", "2015_01", "2016_01", "2017_01", "2018_01", "2019_01")
KEEP_MEMBERS = ("ProviderInfo_Download.csv", "StateAverages_Download.csv",
                "StateUSAverages_Download.csv")

CUTPOINT_CAPTURES = (
    "20160929045708", "20161027025421", "20161124032922", "20161229043052", "20170126045526",
    "20170223044515", "20170323054012", "20170427053340", "20170525054130", "20170629091246",
    "20170727044854", "20170824084621", "20170928201940", "20171026091836", "20171123050243",
    "20180104063549", "20180125053555", "20180308051616", "20180329070750", "20180426063135",
    "20180531072402", "20180628073812", "20180726061600", "20180824123551", "20180927114204",
    "20181025211433", "20181129184724", "20181228082500", "20190131162033", "20190228184852",
    "20190328144714", "20190425145324", "20190523152416", "20190626223606", "20190725111441",
    "20190830092042", "20190926123838", "20191025232745", "20191207082800",
)
GUIDE_CAPTURES = (
    "20120813061551", "20150218093503", "20150304061938", "20160805181307", "20170120233026",
    "20180308051615", "20180426063133", "20180726061559", "20190626223607", "20191010102301",
    "20191025232735", "20200430081250",
)

# group -> {local name under data/raw: url}
GROUPS: dict[str, dict[str, str]] = {
    "cutpoints": {
        "cutpoints/Health-Inspection-Cut-Points-2008-2016.zip":
            f"{CMS_DL}/Health-Inspection-Cut-Points-2008-2016.zip",
        **{f"cutpoints/cutpointstable_{ts}.pdf":
           f"{WAYBACK}/{ts}id_/{CMS_DL}/cutpointstable.pdf"
           for ts in CUTPOINT_CAPTURES},
    },
    "guides": {f"guides/usersguide_{ts}.pdf": f"{WAYBACK}/{ts}id_/{CMS_DL}/usersguide.pdf"
               for ts in GUIDE_CAPTURES},
    "crosswalk": {"crosswalk/ssa_fips_state_county2016.csv":
                  "https://data.nber.org/ssa-fips-state-county-crosswalk/2016/"
                  "ssa_fips_state_county2016.csv"},
    "snapshots": {f"snapshots/{t}": u for t, u in SNAPSHOTS.items()},
}

# SHA-256 on 2026-10-28: files under data/raw, and for snapshots the archive as served.
SHA256: dict[str, str] = {
    "cutpoints/Health-Inspection-Cut-Points-2008-2016.zip":
        "e7a23f3adc7a298cb581be06cea8b1597e554d7896bea1136fc0f006a87619d3",
    "cutpoints/cutpointstable_20160929045708.pdf":
        "c5d750db1726cc4efe7e69f4dcb905eb2b8443c11f39d702c769479d38a9b923",
    "cutpoints/cutpointstable_20161027025421.pdf":
        "8a2bf7a819870f8c3559c830d3cc6d3af808689f03c192e6752206708b45d933",
    "cutpoints/cutpointstable_20161124032922.pdf":
        "7c0dd913c3d2fc53cccd042d8697123c32dd82bcfebe21dac7345b0bdfdf7f3e",
    "cutpoints/cutpointstable_20161229043052.pdf":
        "4499d73fc9b3247210f314ddbf3eabf67c3a3d470c104bc43ea6108555fe8d2a",
    "cutpoints/cutpointstable_20170126045526.pdf":
        "09067c1d9a0bda080ddc16de5bb596ba2e65b9e4982d43c008a9d9bf905efb1f",
    "cutpoints/cutpointstable_20170223044515.pdf":
        "e13f6509b549fd28a1663247fc12931b57a5c86ebe2613adfd727b66fec27348",
    "cutpoints/cutpointstable_20170323054012.pdf":
        "83a6b06d59356d4dc563ad4d9c57374e5ef00dd1fc8e860c0b838bafba4bfcbf",
    "cutpoints/cutpointstable_20170427053340.pdf":
        "cff8dd3c048ab8539c0ea55b8eece9336b84bc4c85f8e75e3aa35bdfb3042194",
    "cutpoints/cutpointstable_20170525054130.pdf":
        "ca19aeeb50d562a6ec5ebe61d0e0c979eb223329c922d38692f41804f932e948",
    "cutpoints/cutpointstable_20170629091246.pdf":
        "b73fde820793d2a0665cf29d1036e20c95d77cf70c0bee0f46f16c074a254cc1",
    "cutpoints/cutpointstable_20170727044854.pdf":
        "75998bb21c14071b35c0e9abf9188de17507fb684c8b308416cbf231a52b3970",
    "cutpoints/cutpointstable_20170824084621.pdf":
        "3ca3581c3b1ef398362588203fdd899c7c97480ae69f843164cab6f16d7adcec",
    "cutpoints/cutpointstable_20170928201940.pdf":
        "dde11cb5f60bc3b6755a2d3254196475b5075586c295c67d09ecbb3b01d5d844",
    "cutpoints/cutpointstable_20171026091836.pdf":
        "4d538b177f8b04449d25d95b8c3b49c75247dcac0547bc4fff2d15ae2d95f547",
    "cutpoints/cutpointstable_20171123050243.pdf":
        "4ce681fadc128c8844e45c21b4759aee4821465b393bbc543e008aa2c37f0ceb",
    "cutpoints/cutpointstable_20180104063549.pdf":
        "cfba37b10d1f36695086487f1a4b3eccf2870a6b55245d69a741b4cfd8fabd09",
    "cutpoints/cutpointstable_20180125053555.pdf":
        "cf8e6ed5a012a1fc9c4c7d0d2f30cbfbba6389d154e0108f7cd35cef847409e8",
    "cutpoints/cutpointstable_20180308051616.pdf":
        "1f428a61a4bde8a0920e760a023ba604bf50a71675bc898c42304d88ff6aa796",
    "cutpoints/cutpointstable_20180329070750.pdf":
        "6d481b389cd04053d61a71ce7526bcc6ff5436654c896979080cffa1d27498d7",
    "cutpoints/cutpointstable_20180426063135.pdf":
        "c291e8e4d34a6c3dacae3099806b24c467318adf9db3f1d6a0042c0cea2f10b9",
    "cutpoints/cutpointstable_20180531072402.pdf":
        "d33a61dba19322f37222fcefefb82d2804d65f1def32682e2f042247dcd0eaf5",
    "cutpoints/cutpointstable_20180628073812.pdf":
        "4f5a386ce1988750857f4865071cb891cfc312a1c428629c0803251b32e153f8",
    "cutpoints/cutpointstable_20180726061600.pdf":
        "8b8c57c6db661d9025ee4aeb7a913dd246b5ae6ba84f1541b09a215e3af05fd0",
    "cutpoints/cutpointstable_20180824123551.pdf":
        "3873d5b330ec4e7bc36e7433a05eb4478a5b9ce63bf8d29b5ecd39aaebdfe535",
    "cutpoints/cutpointstable_20180927114204.pdf":
        "0976bb6a48a864485de6bb96ac5aea6de3b2643e492f17acfe8a2b88c1a5119f",
    "cutpoints/cutpointstable_20181025211433.pdf":
        "651efb973f6c157296dc9cf85d01264708058233cbc3ff14490a9522c27bc19b",
    "cutpoints/cutpointstable_20181129184724.pdf":
        "050e0dd0abc3535ee4674ff9ffe046a78285f985f2fa6cff04c802d7e3f434cf",
    "cutpoints/cutpointstable_20181228082500.pdf":
        "724c540a0480c4f879ed8ee435758d54ba423a96860840ef0c3bba38788f0c42",
    "cutpoints/cutpointstable_20190131162033.pdf":
        "98ba00bb061fcd971ec790acc383aed08ed3dbb87e667d250eb1424141eaa162",
    "cutpoints/cutpointstable_20190228184852.pdf":
        "1cd9f825b1282ba607cf97bba541762cc5f11c5fdf9f318d48f600c2dfa1780b",
    "cutpoints/cutpointstable_20190328144714.pdf":
        "034b9e7aeb9e1b26f14b2386bc5495cd8bc8eb581370a06e0ea8ac4f0e6e2453",
    "cutpoints/cutpointstable_20190425145324.pdf":
        "3d5da26677f3297e794495f34da036ac0fa84b69d06e8b8292faf409720932bb",
    "cutpoints/cutpointstable_20190523152416.pdf":
        "2db1c28feadc44fae8fe3a95bd20d6e5dabb179f979082066f213ac671afe876",
    "cutpoints/cutpointstable_20190626223606.pdf":
        "8aca15fe1c7822d0dc5bac8c791ae8648b0a4cba5f9af48eab1706bc6730a0ba",
    "cutpoints/cutpointstable_20190725111441.pdf":
        "a95b0d3a8740a7e48fa637f073048f3d3c9a2060305a5a77760fc6ebc7b78860",
    "cutpoints/cutpointstable_20190830092042.pdf":
        "5d2fe4c60b6c89b196c027ac16702eb70143e4923cf4a309c685ac10b7f590d2",
    "cutpoints/cutpointstable_20190926123838.pdf":
        "d51dc47bc05ec8233a52c3adc4beea32ed156cdb2ce8b75005f805b9ab40c38f",
    "cutpoints/cutpointstable_20191025232745.pdf":
        "71406170a9f6f77b641f177c1b8af3f862df03e03225d3d48e51b156577972ce",
    "cutpoints/cutpointstable_20191207082800.pdf":
        "c3d6fd3c5d682206720c0564ce481c3125dc609d6733f0d8511291d50ed53fd1",
    "guides/usersguide_20120813061551.pdf":
        "3ba40d29f9944ec17c3b24db8539fcb76cf62273ddda54e33d702e9f4631eb08",
    "guides/usersguide_20150218093503.pdf":
        "8b388738d0540734ffc7a213cc7b4fca6711374014a86c4844665d2db8ad08f1",
    "guides/usersguide_20150304061938.pdf":
        "e647eaf996fd74fef07c244d0f21a2258242aecd49d8063cba4f5f37b2c8117a",
    "guides/usersguide_20160805181307.pdf":
        "aa3e35687faf61728b0592c169971446371c23c9d59a74dfa3e6eff0d6a50633",
    "guides/usersguide_20170120233026.pdf":
        "b82c958f22166ba8d36c01796608ba25c80b17b051d95e5617449722c2cb4a1c",
    "guides/usersguide_20180308051615.pdf":
        "9904f5ece6560086ed9bfedb85f04a46f5f41d4bc4f59adedf5e9e90479b9883",
    "guides/usersguide_20180426063133.pdf":
        "0154ed88adc2e47db887134be74048f80ab12215b41f2ee55802ec9afe84da93",
    "guides/usersguide_20180726061559.pdf":
        "bb374dfcc9deb377470e121c36313fddc90939ac659933ea9e39ea99ea19ae14",
    "guides/usersguide_20190626223607.pdf":
        "9d2c998dd35fe87a7d1ced0f771e2af5f34014c54397bed08b084b1ef978ee7f",
    "guides/usersguide_20191010102301.pdf":
        "ea3d2c91e08765a067b5e7fb2543aa4552026e5f6600a64338bb466eee4f009d",
    "guides/usersguide_20191025232735.pdf":
        "6b9c8ee30ecd77fb702112260d1e3d04cd8317e542f0e6f1631240dc90b4185b",
    "guides/usersguide_20200430081250.pdf":
        "b32fddc892fd3f245941f7504b04648544fb94c15f89bc20cdd56c1f7656e5f1",
    "crosswalk/ssa_fips_state_county2016.csv":
        "1e74250a8a9ccb5b2b19e801f1a3948b84462eee3f86f04208a78087cf046395",
    "snapshots/2013_08":
        "5c5e9dfefc336715574eaa53339d61587f4d4763236323bb401c9ccd98da418e",
    "snapshots/2013_09":
        "79e9a3960320c8f5d332bc960a7ba9cb6ad46efa91c5fa575178da4410ad3e65",
    "snapshots/2013_11":
        "6139c7f3b279bc8ece5257ec79351bf69c751f7f9b05f282f4aba71300adc9af",
    "snapshots/2014_01":
        "d088eb8b80bd389e6a94df10368d7480337cba0733b9397976cd5f729711a9bf",
    "snapshots/2014_02":
        "a68d35cec34becd3f5b2e41ef6591c98a8f00817bffe0df59d0f090e11a635d4",
    "snapshots/2014_03":
        "b013f22262288dc675b032b560f738513fc1af6a983b8abab6bf3d3737afbb42",
    "snapshots/2014_04":
        "d9af417fc70c5e7ba5d2114b91e46788b700ae7aa5453352c8771345925e4e71",
    "snapshots/2014_06":
        "3b8409a703b24f256063dbdeb9a7100b3a34d6b97b28594ac3b3bc5aae542281",
    "snapshots/2014_07":
        "dbb9638069d53a519fdaedff97e595e0fe1a74f624299ef509105a6cd5e94348",
    "snapshots/2014_08":
        "9db37cc4ab8b93676d5f140e452e594f226bf0511e0b06c4c903cefbb95cf85c",
    "snapshots/2014_09":
        "fa713bbd45a80542c6f56bf0526fc099bb4b2b167bc3c84ccb490828c1ed5fa3",
    "snapshots/2014_11":
        "cbf72f74c48f3da796a01844ec012bb91629ff0da2f5e2d3e4ebe4f24e313ec8",
    "snapshots/2014_12":
        "f5d265c788703a2649a23bc98e15a43632e778624faf1a587ec31d079f32eeff",
    "snapshots/2015_01":
        "17539289c891d78b05d03cc44f80447f3201603ca1fc53400897e288ddbb70da",
    "snapshots/2015_02":
        "ded13949cf06ad73b324bcc01a2b9d07e19c78e9e7ac31777a218e114c7fc8f6",
    "snapshots/2015_03":
        "5636e13a0f10c7514a97311f2b6abde03485e089f61b2a5ecf03b9b95895198e",
    "snapshots/2015_04":
        "db31424c5d50709c259685ea4661b0268097148c5edf4f0dc250355f145c39da",
    "snapshots/2015_05":
        "02c0f8fb149f26ec192b06ff80ca2c96320a3124d36753e6052a111aa8af166a",
    "snapshots/2015_06":
        "b629dd25f52e851044ea5a82a4244682254e5dc24547eb2ad2f5e4ed4b868302",
    "snapshots/2015_07":
        "5f9769ef03f5e8a19b5c5966a4339ca0007d47ea83a3c07e030bca2f09bcd634",
    "snapshots/2015_08":
        "0893fad81bca084ee6106f6e7e66426a1f553089b76db80d571ff4ef04a0a5bb",
    "snapshots/2015_09":
        "bd64d35c2bd5f464159fede365b95c2f3e443eba4d77a2ae6bfa5a0f61d56bb5",
    "snapshots/2015_10":
        "cb3f3bc401964c0b45d46f9c6f8b7d88f86bbf6c168d9945229f79587930ddfb",
    "snapshots/2015_11":
        "cf4f679984d82e91c60d3e8bb6e3ac660cad14ca34bac27a1bf75de69ec64d53",
    "snapshots/2015_12":
        "3a5d12a6a79b4083dde01cb09200ad5858d74f05dde0003ac99116d312c39fb8",
    "snapshots/2016_01":
        "3a5d12a6a79b4083dde01cb09200ad5858d74f05dde0003ac99116d312c39fb8",
    "snapshots/2016_02":
        "b589f55300846ef36c92b7dc57d413f8704fd0e8f38e1902b9955c2082b77d2a",
    "snapshots/2016_03":
        "b601f1d570130dd2c86b9932e2d476190c2e27310dacab4effd6184a250f5f87",
    "snapshots/2016_04":
        "b601f1d570130dd2c86b9932e2d476190c2e27310dacab4effd6184a250f5f87",
    "snapshots/2016_05":
        "022d021b8ef993428e41a6d602448b89da759c962862544b929cda030a341468",
    "snapshots/2016_06":
        "c460957ccf3d7b18746668a9050376e46eefce6594e8fd21a1cf14384c76b2b7",
    "snapshots/2016_07":
        "c460957ccf3d7b18746668a9050376e46eefce6594e8fd21a1cf14384c76b2b7",
    "snapshots/2016_08":
        "8a1e025d4a8e529da05c8c5f907b05e6b24f3f6961e625a48065b1c4fe0554cb",
    "snapshots/2016_09":
        "5652297fe290d97cd553351188d0d360113e0febbcef6479ba899aa9660f6ed6",
    "snapshots/2016_10":
        "3c601377e44db90a10fea08c77335bc46977b6f02b28f73574dbc3c9488a8ec7",
    "snapshots/2016_11":
        "3dd8b711876ce6784510f576e5cc696692e8f31afaa99e8ba3fe5d63348d8b9c",
    "snapshots/2016_12":
        "29fe283f124c3c571625b3c0fb48bdc24b8ac32a01110b3956e2a7fc534f9ba1",
    "snapshots/2017_01":
        "29fe283f124c3c571625b3c0fb48bdc24b8ac32a01110b3956e2a7fc534f9ba1",
    "snapshots/2017_02":
        "7b43256904725a507cd6cb28db51a72c41d5e5b40e42f295f79ef86a3c563d5a",
    "snapshots/2017_03":
        "16d5dc4b2b742153a0729a377398555ff3d8bdce6792ec4f2be443b3eb6adfe2",
    "snapshots/2017_04":
        "16d5dc4b2b742153a0729a377398555ff3d8bdce6792ec4f2be443b3eb6adfe2",
    "snapshots/2017_05":
        "0aaba250399aa8ae5f8bf308db9917c32c6beed188c25bf9ebe63e371b3e378a",
    "snapshots/2017_06":
        "0aaba250399aa8ae5f8bf308db9917c32c6beed188c25bf9ebe63e371b3e378a",
    "snapshots/2017_07":
        "875b90541923539ebf41211ab16984b9defee046284ff3aa4ff9bba95dc61ab6",
    "snapshots/2017_08":
        "3099271b5fc20f49ce0f3cd37bc76ce95119e2d7fbba6d111d8fee6b069b0275",
    "snapshots/2017_09":
        "3099271b5fc20f49ce0f3cd37bc76ce95119e2d7fbba6d111d8fee6b069b0275",
    "snapshots/2017_10":
        "60ed67b2dda74202db51b9a9405fb431a25380e7d59dde13e0de65f35506e475",
    "snapshots/2017_11":
        "a74566c42fdd6d945452a66f921728cd4b5427056ec2a0b561488895d708a39f",
    "snapshots/2017_12":
        "aad0dc640ac868c4232251699df306169696f6798a50664d6cbf33294e07558a",
    "snapshots/2018_01":
        "da11f3f5fbb9d9c81ae97154b96d71c177da5aa5524458b668700b26690c39f4",
    "snapshots/2018_02":
        "da11f3f5fbb9d9c81ae97154b96d71c177da5aa5524458b668700b26690c39f4",
    "snapshots/2018_03":
        "932fb5a5744c8d74d60b5e138636e4b27c1fb2d05bdc3f16a22e575e1ed86232",
    "snapshots/2018_04":
        "dc30dbc43bb59a91721bd1788d20ddadd9e1f6bb20f4991e7479f4d7eb2d9d25",
    "snapshots/2018_05":
        "f0db2636dc1adc40b0ac99d18f0add42fcb0abd640898a6e16ce9a85ac8ff215",
    "snapshots/2018_06":
        "61bce9a27301780278c848e019326ac4f53b0e041bcd8b6c89a0ed88a3f987d9",
    "snapshots/2018_07":
        "61d582c3ef757baa2258a8e73de54e5a9e1304c9b42b363db33c2792f4ac55ef",
    "snapshots/2018_08":
        "19f414eca27ef4c22d559624c782d9ba7c32f132d449b73516a00f837f71e0cc",
    "snapshots/2018_09":
        "19f414eca27ef4c22d559624c782d9ba7c32f132d449b73516a00f837f71e0cc",
    "snapshots/2018_10":
        "88c827f5ac7f7d6294f63ff95631262592a67ffd11a2ed102966c43ece40bdbe",
    "snapshots/2018_11":
        "84409975ce790624f46420e33efdfb03b3399c56613ced75e97f7300c40d0f69",
    "snapshots/2018_12":
        "82c86eefdc1c3518aac3a2b6a63edcbbddd8fe33553c88e01fdec20fd60bf836",
    "snapshots/2019_01":
        "82c86eefdc1c3518aac3a2b6a63edcbbddd8fe33553c88e01fdec20fd60bf836",
    "snapshots/2019_02":
        "877bf31f27ba0b70a6ef0dc72ef70840b6d6e992c19c75bd280625a6923a837f",
    "snapshots/2019_03":
        "544da8dcf51c54daf4ebb8fb61760dc9d6c70d1cf34bed1acefc0336c5d36c0c",
    "snapshots/2019_04":
        "1aa8532b1fe74464a5d160652a0d43b8c7d8915949bf557f0824df34c583a34b",
    "snapshots/2019_05":
        "1aa8532b1fe74464a5d160652a0d43b8c7d8915949bf557f0824df34c583a34b",
    "snapshots/2019_06":
        "82e84fbc4df444c67e2f2ea95b80cb587cbbb9c00a6369aed100b95941e99e75",
    "snapshots/2019_07":
        "72e0be7023af2e7125e9869765661b80625f88d8811825cbccb78704fa3109db",
    "snapshots/2019_08":
        "72e0be7023af2e7125e9869765661b80625f88d8811825cbccb78704fa3109db",
    "snapshots/2019_09":
        "9b17de7fd4e0190e33cb579809bf35edc884889af27186d9d32145680585cd26",
    "snapshots/2019_10":
        "95d57b4d43683a3286b42b72820cad21fd084c96400f9d86e1dbf77ed228416c",
    "snapshots/2019_11":
        "f1d4a43967f2fca5363ee33dc13e7e54666179845a1e6c90648f9179a709a304",
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _get(url: str) -> bytes:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                    timeout=600) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{url}: HTTP {e.code}. If this is 403 or a challenge page, the "
                           "source blocks scripts; do not work around it.") from e


def _write_atomic(dest: Path, body: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(body)
    os.replace(part, dest)


def fetch(url: str, dest: Path) -> int:
    if dest.exists() and dest.stat().st_size > 0:
        return dest.stat().st_size
    body = _get(url)
    if body[:15].lower().startswith((b"<!doctype html", b"<html")):
        raise RuntimeError(f"{url}: got an HTML page, not the file (a challenge page?)")
    _write_atomic(dest, body)
    return len(body)


def open_archive(body: bytes) -> zipfile.ZipFile:
    """The NBER copies from 2015/07 on are a zip inside a gzip stream."""
    if body[:2] == b"\x1f\x8b":
        body = gzip.decompress(body)
    return zipfile.ZipFile(io.BytesIO(body))


def fetch_snapshot(tag: str, url: str, keep_archive: bool = False) -> tuple[int, str]:
    """Download one monthly archive, keep the members listed in KEEP_MEMBERS (and the
    ownership file for OWNERSHIP_TAGS) as ``<member>.gz`` in snapshots/<tag>/, record the
    archive's size and SHA-256 in snapshots/manifest.csv. Returns (bytes, sha256)."""
    out = SNAP / tag
    done = out / "ProviderInfo_Download.csv.gz"
    manifest = _manifest()
    if done.exists() and done.stat().st_size > 0 and tag in manifest:
        return int(manifest[tag]["bytes"]), manifest[tag]["sha256"]
    body = _get(url)
    if body[:15].lower().startswith((b"<!doctype html", b"<html")):
        raise RuntimeError(f"{url}: got an HTML page, not the archive")
    digest = sha256_bytes(body)
    out.mkdir(parents=True, exist_ok=True)
    if keep_archive:
        _write_atomic(out / "archive.zip", body)
    z = open_archive(body)
    want = set(KEEP_MEMBERS) | ({"Ownership_Download.csv"} if tag in OWNERSHIP_TAGS else set())
    names = {Path(n).name: n for n in z.namelist()}
    if "ProviderInfo_Download.csv" not in names:
        raise RuntimeError(f"{url}: no ProviderInfo_Download.csv in the archive")
    for short, full in names.items():
        if short in want:
            _write_atomic(out / f"{short}.gz", gzip.compress(z.read(full), mtime=0))
    _manifest_add(tag, url, len(body), digest, sorted(names))
    return len(body), digest


def _manifest() -> dict[str, dict]:
    p = SNAP / "manifest.csv"
    if not p.exists():
        return {}
    with open(p, newline="") as f:
        return {r["tag"]: r for r in csv.DictReader(f)}


def _manifest_add(tag: str, url: str, size: int, digest: str, members: list[str]) -> None:
    rows = _manifest()
    rows[tag] = {"tag": tag, "url": url, "bytes": size, "sha256": digest,
                 "downloaded": DOWNLOADED, "members": ";".join(members)}
    SNAP.mkdir(parents=True, exist_ok=True)
    tmp = SNAP / "manifest.csv.part"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag", "url", "bytes", "sha256", "downloaded",
                                          "members"])
        w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])
    os.replace(tmp, SNAP / "manifest.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="+", choices=list(GROUPS), default=list(GROUPS))
    ap.add_argument("--check", action="store_true", help="list sizes and hashes only")
    ap.add_argument("--keep-archives", action="store_true",
                    help="also keep each snapshot archive (about 1.9 GB in all)")
    args = ap.parse_args()
    total = 0
    for g in args.only:
        for name, url in GROUPS[g].items():
            note = ""
            if g == "snapshots":
                tag = name.split("/")[1]
                if args.check:
                    m = _manifest().get(tag)
                    size, got = (int(m["bytes"]), m["sha256"]) if m else (0, "")
                else:
                    size, got = fetch_snapshot(tag, url, args.keep_archives)
            else:
                dest = RAW / name
                size = (dest.stat().st_size if dest.exists() else 0) if args.check \
                    else fetch(url, dest)
                got = sha256(dest) if size else ""
            if size:
                want = SHA256.get(name)
                if want is None:
                    note = f"  sha256 {got}"
                elif got != want:
                    note = f"  [sha256 differs from {DOWNLOADED}: {got[:12]}...]"
            total += size
            print(f"{size:>12,d}  {url}{note}", flush=True)
    print(f"{total:>12,d}  total bytes")


if __name__ == "__main__":
    main()
