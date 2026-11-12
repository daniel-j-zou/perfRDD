# NC DAC Offender Public Information bulk files (sentencing RDD feasibility)

Fetched 2026-11-11. Listing page: https://webapps.doc.state.nc.us/opi/downloads.do?method=view
Files: https://opus.doc.state.nc.us/offenders/<CODE>.zip

## Authorization and robots.txt
The host opus.doc.state.nc.us serves `robots.txt` with `User-agent: * / Disallow: /`.
The project author explicitly instructed this download: **"Yes, fetch them"** (2026-11-11),
as they did earlier for ftp.cdc.gov. Each file was fetched once with curl (no crawling,
5 s pause between files). All content is untrusted data: it was only unzipped and parsed
by `inspect_layout.py` (run with `python -I`), never executed.

## Scripts and outputs
- `download.py` (idempotent: skips a file whose size equals Content-Length; unzips to `data/raw/<CODE>/`), log in `download.log`.
- `inspect_layout.py` writes `layouts.txt` (all field lists) and prints the report saved in `inspect_output.txt`.
- `data/raw/` is git-ignored (`git check-ignore` confirmed, rule in `experiments/.gitignore`). It holds the zips plus the
  extracted files, about 12 GB in all (the SC .dat alone is 4.5 GB); the disk had 8.6 GB free afterwards. The zips can be deleted to save about 0.8 GB.
- INMT4AA1 and OFNT3BB1, APPT7AA1 contain names of real people (public records); never print or commit them.

## Files (bytes from server Content-Length; extracted .dat size)
| Code | Table | zip bytes | .dat MB |
|---|---|---|---|
| PublicTables.pdf | structure diagram | 9,106 | |
| OFNT3AA1 | Offender Profile | 41,296,982 | 559 |
| APPT7AA1 | P&P Client Profile | 48,349,547 | 471 |
| INMT4AA1 | Inmate Profile | 47,501,029 | 521 |
| INMT4BB1 | Sentence Computations | 21,525,984 | 178 |
| OFNT3BB1 | Court Commitment | 139,373,094 | 1,231 |
| OFNT3CE1 | Sentence Component (SC) | 318,405,389 | 4,549 |
| OFNT3DE1 | Special Conditions and Sanctions | 34,748,442 | 320 |
| INMT4CA1 | Parole Analyst Review | 29,926,321 | 615 |
| INMT9CF1 | Disciplinary Infractions | 96,234,213 | 2,164 |
| OFNT1BA1 | Financial Obligation | 67,208,413 | 1,200 |
| OFNT9BE1 | Warrant Issued | 5,445,634 | 51 |
| APPT9BJ1 | Impact Scheduling Request | 87,104 | 0.5 |

Each zip holds one `<CODE>.dat` (fixed width, no delimiter, one line per record, `?` runs = missing,
`0001-01-01` = missing date) and one `<CODE>.des` (Name, Description, Type, Start (1-based), Length).
Data are current to early Nov 2026 (latest conviction date 2026-11-06). The tables hang off the offender
id; SC joins to commitments and computations by (id, commitment prefix, sentence component number).

## Key fields (exact names)
**Offender id**: `CMDORNUM` (OFNT3AA1, OFNT3BB1, OFNT3CE1, OFNT3DE1, OFNT1BA1, OFNT9BE1), `CIDORNUM` (INMT4AA1, INMT4BB1,
INMT4CA1, INMT9CF1), `CDDORNUM` (APPT7AA1, APPT9BJ1); 7 characters. Sentence key: `CMPREFIX`/`CIPREFIX` + `CMSENCMP`/`CISENCMP`.

**Sentence Component OFNT3CE1** (one row per sentence component):
- prior record: `PRLEVEL` (LEVEL I..VI) and `PRPOINTS` ("PRIOR RCD. POINTS/CONVICTIONS", 3 chars)
- offense class: `PENCLASS` (CLASS A, B1 SS, B2 SS, C..I, J FAIR SENTENCING, CLASS 1/2/3/A1 MISDEMEANOR SS, CRV, POST RELEASE REVOCATION, PARTIAL REV CONFINE, old classes "FELONS (PREFAIR)", "MISD.(PRE-STRUCTURE)"); `PRIFELMD` felony/misdemeanor
- punishment type: `PUNISHTYP` (ACTIVE SS, INTERMEDIATE SS, COMMUNITY SS (DCC), DWI, POST RELEASE, PRE-SS (FAIR) DCC, FAIR FELONS, FAIR MISDEMEAN, PRE-FAIR, NON-N.C. OFF., NON JUDGMENT CASES, PSI); `CMSNTYPE` (sentence type, 1 of 6 columns CMSNTYPE..CMSNTYP6: PROBATION, "DEPT OF CORR DIV OF PRISONS", ...)
- terms: `CMTRMMIN`, `CMTRMMAX`, `CMTRMPRB` (supervision), `SSACTTRM` (split-sentence active term)
- dates: `CMIMPSDT` (sentence conviction date), `CMSENTDT` (sentence effective date), `CMOFFDTE`/`CMOFFDT2` (offense begin/end)
- county: `CMCOCNVT`; offense: `CMOFFNSE` (code and text together, e.g. "LARCENY (OVER $200)"), `CMSTATUT` (statute number), `CMOFFQLF`
- status: `CMCMPSTA` (ACTIVE, CORRECT, AMENDED, VACATED, ...; 2.9M blank), `CMCMSTDT`, `CMPPCMST`; also `CMCHNTYP`, `CMDISPOS`, `COURTTYP`, `CMCOUNTS`

**Court Commitment OFNT3BB1**: `CMPREFIX`, `CMADMDTE` (offender admission/intake date), `CMINTDTE` (P&P case intake date),
`CMNEWPD` (new period of incarceration flag Y/N), `CMNEWSUPV` (new period of supervision flag), `GMSEFFDT`, `CMSTAFLG`/`CMSTADTE`, `CMPRIOFF` (most serious offense), `CMCOUNTY`, `CMMAXLEN`. There is no commitment type field.

**Inmate Profile INMT4AA1** (one row per inmate, current/latest incarceration only): `CIRADMDT` (admission date), `CICONVDT`, `CIPRIOFF`, `GIRPED`/`GIRPRD`/`GIRMAX` (release dates), `PRIORSFL` (prior incarcerations flag), `CIDYSERV`, `PTSTATUS`. No admission-type field.
**Sentence Computations INMT4BB1** (key CIDORNUM+CIPREFIX+CISENCMP): `CISTAFLG`, `GIEFFDT` (sentence begin), `CIRELDAT` (actual sentence end date), `GISPRD` (projected release), `PARDISCHDT`, `PARBEGDT`. No admission type.
**P&P Client Profile APPT7AA1** (one row per client): `PPRCDSTA`, `GDSTATUS`, `PRIOFFNSE`, `SNTPUNTY` (sentence punishment type), `DCCPUNTYP`, `CDSUPLTH` (overall supervision length), `DTOFUPDT`. No supervision start or end date and no termination reason (the latter only via `CMPPCMST`/`CMPPSTDT` in SC and `CMPPSTFL`/`CMPPCMDT`/`CMINTDTE` in OFNT3BB1).
**Offender Profile OFNT3AA1**: `CMCLBRTH` (birth date), `CMCLSEX`, `CMCLRACE`, `CMETHNIC`, `CMCITIZN`, `CMBRTHST`, `CMBRTNCC`, height/weight/hair/eye. No education field. (Race/sex/birth date also in INMT4AA1 and APPT7AA1.)
Other tables: disciplinary (INMT9CF1), financial obligations (OFNT1BA1), special conditions (OFNT3DE1), warrants (OFNT9BE1): not needed for the design.

## Counts (from `inspect_output.txt`)
- SC: 4,849,557 records, 1,292,340 distinct offenders. Conviction date `CMIMPSDT` runs 1900-02-01 to 2026-11-06 (10 missing; rows before 1970 are junk/legacy); effective date `CMSENTDT` 1900-01-06 to 2083-09-30.
- `PUNISHTYP` counts (all years): COMMUNITY SS (DCC) 1,342,678; ACTIVE SS 708,403; INTERMEDIATE SS 691,786; DWI 633,369; PRE-SS (FAIR) DCC 417,529; POST RELEASE 326,681; FAIR MISDEMEAN 253,345; PRE-FAIR 216,063; FAIR FELONS 195,558; NON-N.C. OFF. 48,999. (The value is "ACTIVE  SS" with two spaces.)
- `PRPOINTS` is populated on every row (zero on 2,549,043; nonzero on 2,300,514). `PRLEVEL` non-missing on 3,108,560 rows.
- Post-2011 (conviction on or after 2011-12-01, proxy for the Justice Reinvestment grid, which applies by offense date): 1,316,934 rows, of which 393,390 are felony SS rows (class B1..I) with a level and a punishment type of ACTIVE/INTERMEDIATE/COMMUNITY. Share ACTIVE SS, felony class (rows) by `PRLEVEL` I..VI:

| class | I | II | III | IV | V | VI |
|---|---|---|---|---|---|---|
| B1 | 1.00 | .99 | .99 | 1.00 | 1.00 | 1.00 |
| B2 | .98 | .98 | 1.00 | 1.00 | 1.00 | 1.00 |
| C | .88 | .91 | .96 | .99 | 1.00 | 1.00 |
| D | .76 | .90 | .96 | .99 | .99 | 1.00 |
| E | .15 | .18 | **.73** | .97 | .99 | .99 |
| F | .22 | .19 | .22 | **.72** | .83 | .88 |
| G | .10 | .10 | .13 | .26 | **.88** | .96 |
| H | .06 | .07 | .10 | .21 | .28 | **.91** |
| I | .03 | .03 | .04 | .13 | .18 | .21 |

  Cell sizes are large (class H: 48k, 53k, 35k, 15k, 9k, 4k; class I: 26k, 35k, 25k, 9k, 4k, 6k; class G: 9k, 17k, 13k, 6k, 1k, 1k).
- The jump is visible in the points themselves (active share at PRPOINTS, n in brackets): class E 5: .24 [954], 6: .81 [318]; F 9: .26 [1238], 10: .66 [564]; G 13: .24 [1080], 14: .81 [392]; H 17: .29 [1665], 18: .86 [675]. 94.1% of felony rows have PRPOINTS inside the band implied by PRLEVEL (I 0-1, II 2-5, III 6-9, IV 10-13, V 14-17, VI 18+); outliers (max 512, 610) are data errors.
- Misdemeanor rows carry PRLEVEL I-III (levels by conviction count; PRPOINTS then holds a count).
- Commitments OFNT3BB1: 3,995,463 rows; `CMNEWPD` Y 992,701, N 2,427,678, blank 575,084; `CMADMDTE` valid on only 992,795 rows (about the Y rows), `CMINTDTE` on 3,483,897. 57% of offenders have 2+ commitments. 209,334 offenders have a post-2011 felony SS row; 84,315 of them have a commitment admitted (valid `CMADMDTE`) more than 30 days after the first such conviction (crude, includes revocation commitments).
- Id matching (SC ids found in table / table ids found in SC): OFNT3AA1 1.000 / 0.993; OFNT3BB1 1.000 / 1.000; APPT7AA1 0.925 / 0.995; INMT4AA1 0.391 / 0.996; INMT4BB1 0.393 / 1.000. Distinct ids: INMT4AA1 507,152; INMT4BB1 507,898; APPT7AA1 1,201,623; OFNT3AA1 1,300,675; OFNT3BB1 1,292,419. 408,578 ids are in both Inmate and P&P profiles. The low 0.39 is expected: Inmate Profile holds only people with a prison record; probation-only offenders are absent.

## Assessment: is the design feasible?
Yes for the first stage; the outcome needs care.
1. **Prior record level at sentencing**: yes, `PRLEVEL` (and the points) sit on each sentence component. `PRPOINTS` is a numeric points field for felonies, so a points running variable with cutoffs 1|2, 5|6, 9|10, 13|14, 17|18 is available, 94% consistent with the level. Caveats: the field is "points/conviction count", not labelled by type; the post-2011 classification is by conviction date (not offense date, `CMOFFDTE` can refine it); the points are the court-recorded value, and may be updated after the fact.
2. **Punishment type per sentence**: yes, `PUNISHTYP` per (id, prefix, component), and the grid pattern is sharp (class E jumps from .24 to .81 at 5|6 points). Caveats: `PUNISHTYP` mixes the sentence structure with the supervision stream (POST RELEASE, CRV, DWI, revocation classes); it records the imposed punishment, not the grid cell (A, I/A, C/I/A), which must be rebuilt from class and level using the statutory grid. "ACTIVE SS" is the prison sentence; intermediate (supervised probation) and community are the probation side. Convictions with several counts and consecutive sentences yield several rows per offender: use one index sentence per case (`CMCASENO`, `CMIMPSDT`).
3. **Later new commitment by id**: yes, the id is shared (SC, commitments, probation and inmate profiles all match at 99% or higher from the table side). Later commitments can be found in OFNT3BB1 (`CMNEWPD`='Y', `CMADMDTE`) and later SC rows. Revocations are not directly typed in the commitment table, but SC marks them through `PENCLASS` (CRV, POST RELEASE REVOCATION, PARTIAL REV CONFINE) and `PUNISHTYP` (POST RELEASE), so "new offense commitment" versus "revocation" can be separated by requiring a new conviction date (`CMIMPSDT`) after the index sentence. Reincarceration after release is observable only as dated commitments; INMT4AA1 shows just the current stay, and INMT4BB1 gives release dates by sentence.
4. **Missing or weak**: (a) no admission-type field in INMT4AA1 or INMT4BB1 (see the revocation workaround above); (b) APPT7AA1 has no supervision start/end dates or termination reason (only status codes and update date; use SC `CMPPCMST`/`CMPPSTDT` and OFNT3BB1 `CMINTDTE`); (c) no education field in OFNT3AA1; (d) no judge identity, attorney, plea or charge-bargaining information; (e) only offenders ever under DAC supervision or custody appear, so jail-only misdemeanor sentences and acquittals are absent; (f) the files are a current snapshot, so people who died, were released long ago or were expunged may be missing, and state-only recidivism ignores out-of-state arrests; (g) prior record points may be manipulated through plea bargaining (run a density test at each cutoff; counts drop from 954 at 5 points to 318 at 6 in class E, but part of that is the natural decline of the points distribution and it is not tested here); (h) the grid is fuzzy: above the cutoff the share active is 0.7-0.9, not 1, because the I/A and C/I/A cells let the judge choose, so this is a fuzzy RDD with first stage about 0.5-0.6.
