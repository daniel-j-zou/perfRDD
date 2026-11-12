# Brazil VLBW (SINASC births + SIM infant deaths): feasibility of a public-data link

Status 2026-11-12: probe only, nothing committed. Verdict: **no exact link from public data; only a lossy probabilistic link.**

## Sources and robots.txt
- DATASUS FTP `ftp.datasus.gov.br/dissemin/publicos/` (SINASC/NOV/DNRES/DN<UF><year>.dbc, final to 2022; SIM/CID10/DORES/DO<UF><year>.dbc, to 2024; SIM/CID10/DOFET/DOINF<yy>.dbc, national infant deaths). FTP has no robots.txt (`curl ftp://.../robots.txt`: "The file does not exist").
- OpenDataSUS (dadosabertos.saude.gov.br; opendatasus.saude.gov.br redirects there). CSV zips on `s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SINASC/csv/SINASC_2018_csv.zip` (119.6 MB) and `.../SIM/csv/Mortalidade_Geral_2018_csv.zip` (71.7 MB), national. robots.txt: opendatasus returns the HTML app page (no robots file); datasus.saude.gov.br: "User-agent: *  Disallow: /wp-admin/"; S3 hosts return 403 AccessDenied for /robots.txt (none published). Only the first 60 KB of each CSV zip was fetched, to read the header; the full CSVs were not downloaded.
- Nothing fetched from data.nber.org. Raw files are in git-ignored `data/raw/` (`git check-ignore`: experiments/.gitignore:2 `datasets/*/data/raw/`).
- The .dbc format is the only one used for data (CSV headers identical). `dbc.py` is a pure-Python DBC decoder (no package installed).

## Dictionary evidence on NUMERODN
- SIM structure file `Estrutura_SIM_para_CD.pdf` (updated 12/2019) lists it: "Número da Declaração de Nascido Vivo   NUMERODN   Caracter 8   Números", in the block with PESO, SEMAGESTAC, GRAVIDEZ, OBITOPARTO.
- SIM `Estrutura_do_SIM_2025.pdf` (updated 07/2025) and OpenDataSUS `Dicionario_SIM_2025.pdf`: no NUMERODN line.
- SINASC `Estrutura_SINASC_para_CD.pdf` (old): "Declaração 01 NUMERODN C(08) Número da DN, seqüencial por UF informante e por ano." OpenDataSUS `SINASC - Estrutura.pdf` (current): starts at "CONTADOR  C 8  Número identificador do registro"; no NUMERODN.
- **Actual files**: DNAC2018/2019 (SINASC, 60 fields), DOAC2018/2019/2020 and DOINF17/18/19 (SIM, 87-88 fields), and the CSV headers of the national 2018 SINASC and SIM zips contain NO NUMERODN and no other shared record id. Only NUMEROLOTE and CONTADOR exist, per-system.

## Linkage probe: Acre, births 2018
SINASC AC 2018 (by residence): 16,543 births. SIM AC (by residence) 2018+2019: 8,192 deaths, 532 infant (IDADE unit digit 0-3; digit 4 = years); 283 of those were born in 2018 (DTNASC). Infant mortality 17.1 per 1000.
Infant-death block, cohort born 2018 (n=283), usable: DTNASC 100%, SEXO 98.2%, PESO 84.8%, IDADEMAE 88.3%, SEMAGESTAC 78.8%, GRAVIDEZ 86.6%.
Exact-key match of the 283 deaths to SINASC births (CODMUNRES trimmed to 6 digits):

| key | deaths with key | 0 match | 1 match | >1 | share of SINASC keys unique |
|---|---|---|---|---|---|
| DTNASC+SEXO | 278 | 0 | 0 | 278 | 0.000 |
| +PESO | 236 | 36 | 198 | 2 | 0.931 |
| +IDADEMAE | 246 | 22 | 91 | 133 | 0.409 |
| +PESO+IDADEMAE | 227 | 64 | 163 | 0 | 0.996 |
| +PESO+IDADEMAE+CODMUNRES | 227 | 78 | 149 | 0 | 0.999 |
| +PESO+CODMUNRES | 236 | 53 | 182 | 1 | 0.986 |

So DTNASC+SEXO+PESO gives a unique hit for 198 of 283 deaths (70%); adding mother's age lowers coverage (163, 58%) with no ambiguity. No ground truth exists to measure false matches. Deaths with PESO <1500 in SIM: 85; SINASC births <1500: 178 (1.08%). Because deaths are matched to births, survivors are the non-matched births, so unmatched deaths are misclassified as survivors (30% of deaths for the weight key), which attenuates the VLBW death effect, most severely for weight-missing deaths (15%).

## SINASC variables, AC 2018 (n=16,543), usable share
PESO 99.1%, GESTACAO 96.3%, SEMAGESTAC 96.3%, GRAVIDEZ 99.7%, SEXO 100%, IDADEMAE 100%, ESCMAE 98.1%, QTDFILVIVO 85.5%, QTDGESTANT 83.6%, PARIDADE 100%, CONSULTAS 99.9%, CONSPRENAT 97.5%, MESPRENAT 77.0%, APGAR1 94.1%, APGAR5 94.1%, RACACOR 96.5%. Weight heaping: 15.3% of weights are multiples of 100 g (1% expected); check near 1500 g before an RD.

## National sizes
- Infant deaths in SIM DOINF (non-fetal, <1 year): 36,223 (2017), 35,864 (2018), 35,293 (2019); PESO present 88.8-89.6%; PESO <1500 about 15,000 per year (15,232; 15,137; 14,957); NUMERODN absent in all three.
- Births per year and births <1500 g: NOT measured (the 153 MB DNBR2017.dbc download had not finished; DNBR exists only to 2017, later years need 27 per-UF files, roughly 150 MB per year). AC share of VLBW is 1.08%, which is not a national figure.

## Verdict and cost
A national exact-linked file is not feasible from public data. A probabilistic link (DTNASC+SEXO+PESO, optionally IDADEMAE/municipality) covers about 60-70% of infant deaths in AC, with an unverifiable false-match rate. Cost: about 150 MB per birth year (27 SINASC .dbc) plus 4-5 MB per year (DOINF) in downloads, about 2 GB uncompressed per year; a few years fits well under 2 GB. An exact link would need the restricted identified data from the Ministry of Health (CGIAE). Brazilian studies that link use that restricted access.

## Files
`dbc.py`, `probe_ac.py`, `national.py` (run with `python -I`; `--doinf` for the death files). Raw: `data/raw/dbc/`, `data/raw/docs/`.
