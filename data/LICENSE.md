# Dataset attribution and license notice

**Attribution:** Brazilian E-Commerce Public Dataset by Olist — Olist and
collaborators, published on Kaggle as **`olistbr/brazilian-ecommerce`**.

Original source: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

The original Olist data is distributed under **Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**.
This order-level teaching adaptation (`olist_orders.csv.gz`) is supplied under
**the same CC BY-NC-SA 4.0 terms**. A software-code license elsewhere in the
repository does **not** override the dataset's separate license.

License deed: https://creativecommons.org/licenses/by-nc-sa/4.0/

Authoritative legal text: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en

## Changes made for this adaptation

The maintainers selected delivered orders from the existing historical analyst
export, joined purchase-time item/payment/customer/product/seller attributes,
aggregated one row per order, derived a binary late-delivery outcome, removed raw
outcome columns, reviews and personal/unused fields, sorted the records, and
compressed the result. `README.md` describes the exact selection and aggregation;
`metadata.json` binds the released derivative to source CSV hashes and its own
compressed-file SHA-256. No new orders or outcomes were synthesized.

## Required conditions (summary, not a replacement for the legal text)

- **Attribution (BY):** credit the original Olist dataset and source, link to the
  license, and indicate changes. Do not imply endorsement.
- **NonCommercial (NC):** this notice does not grant permission for uses primarily
  intended for or directed toward commercial advantage or monetary compensation.
- **ShareAlike (SA):** if you distribute an adaptation, distribute it under the
  same license or a license allowed by CC BY-NC-SA 4.0, preserving attribution.
- Do not impose additional restrictions that prevent recipients exercising the
  rights granted by the license. No warranties are given.

**Commercial-course caveat:** educational use is not automatically noncommercial.
Inclusion in a paid course, paid training service, or another commercial offering
requires a separate licensing assessment and may require authorization from the
relevant rights holder. This package supplies no separate commercial permission;
obtain permission or replace the dataset with appropriately licensed real data
before using it commercially.

The local source export did not include an original license-text file. This is an
explicit attribution/license notice, **not** a claimed verbatim copy of the complete
CC legal code. Consult the authoritative legal text linked above for the complete
terms, exceptions, disclaimers and rights. Retain this notice with redistribution.
