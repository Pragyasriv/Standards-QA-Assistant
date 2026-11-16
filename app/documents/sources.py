"""Official RoHS and REACH documents selected for this corpus.

The CELEX numbers below were checked on 13 November 2026:

- RoHS consolidated text 02011L0065-20260701 is the EUR-Lex document dated
  01/07/2026. No later RoHS consolidation was published on the Publications
  Office CELLAR for the dates checked through 13 November 2026.
- REACH consolidated text 02006R1907-20260622 is the text linked from the
  ECHA REACH legislation page and the latest REACH consolidation found on
  CELLAR. No later REACH consolidation was published for the dates checked
  through 13 November 2026.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class OfficialSource:
    """A consolidated legal act to download from the Publications Office."""

    document_id: str
    regulation_name: str
    regulation_number: str
    celex: str
    source_url: str
    collection: str
    eur_lex_pdf_url: str
    title_token: str


OFFICIAL_SOURCES: tuple[OfficialSource, ...] = (
    OfficialSource(
        document_id="02011L0065-20260701",
        regulation_name="RoHS",
        regulation_number="2011/65/EU",
        celex="02011L0065-20260701",
        source_url="https://eur-lex.europa.eu/eli/dir/2011/65/2026-07-01/eng",
        collection="rohs",
        eur_lex_pdf_url=(
            "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/"
            "?uri=CELEX:02011L0065-20260701"
        ),
        title_token="2011/65",
    ),
    OfficialSource(
        document_id="02006R1907-20260622",
        regulation_name="REACH",
        regulation_number="(EC) No 1907/2006",
        celex="02006R1907-20260622",
        source_url=(
            "https://eur-lex.europa.eu/legal-content/EN/TXT/"
            "?uri=CELEX:02006R1907-20260622"
        ),
        collection="reach",
        eur_lex_pdf_url=(
            "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/"
            "?uri=CELEX:02006R1907-20260622"
        ),
        title_token="1907/2006",
    ),
)

ECHA_REACH_LEGISLATION_URL = "https://echa.europa.eu/regulations/reach/legislation"
MANIFEST_RELATIVE_PATH = "data/manifest/documents.json"
