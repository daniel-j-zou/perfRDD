# Yelp half-star rounding (waiting for the author's download)

| Field | Value |
|---|---|
| **Q** | a business's average review rating at a date, rebuilt from its review history |
| **Threshold** | the displayed rating is Q rounded to the nearest half star, so the cutoffs are at x.25 and x.75 |
| **Treatment** | the displayed half-star level |
| **Rule class** | several cutoffs on one score (bands) |
| **X** | category, price range, city, attributes, review count |
| **Y** | reviews or check-ins in the following months |
| **n** | about 150,000 businesses |
| **Source** | Yelp Open Dataset (terms of use) |

## Getting the data (author)

The download requires accepting Yelp's dataset terms in a form, which needs your
approval.

1. Go to https://business.yelp.com/data/resources/open-dataset/.
2. Accept the terms and download the JSON archive (several GB).
3. Save it into `data/raw/`.

## Caveats

- Q is a running average that moves with every review, so the timing of the
  rounding must be rebuilt from the review dates.
- Rating counts are small for many businesses.
