"""Resolve official consolidated PDFs from the Publications Office CELLAR."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.documents.errors import DocumentCollectionError

USER_AGENT = "StandardsQA-Assistant/0.1 (official regulatory document collection)"
PUBLICATIONS_BASE = "https://publications.europa.eu/resource"
PDF_ACCEPT = "application/pdf;type=pdfa2a"
REQUEST_TIMEOUT_SECONDS = 180

_REFERENCE_RE = re.compile(
    r"(?P<celex>0\d{4}[A-Z]\d+)\s*[—–-]\s*EN\s*[—–-]\s*"
    r"(?P<day>\d{2})\.(?P<month>\d{2})\.(?P<year>\d{4})\s*[—–-]\s*"
    r"(?P<version>\d{3}\.\d{3})"
)
_TITLE_RE = re.compile(
    r"expression_title[^>]*>(.*?)</",
    re.IGNORECASE | re.DOTALL,
)
_PDF_RESOURCE_RE = re.compile(
    r"https?://publications\.europa\.eu/resource/consolidation/\S+?\.pdfa2a"
)
_FILENAME_RE = re.compile(r"([A-Z0-9]+\.\d+\.pdf)\s+-->")


@dataclass(frozen=True)
class ResolvedDocument:
    """Publication metadata read from CELLAR for one CELEX expression."""

    title: str
    consolidation_reference: str
    version_date: date
    consolidation_version: str
    official_filename: str
    download_url: str


def normalize_whitespace(value: str) -> str:
    """Collapse spaces, including non-breaking spaces, to single ASCII spaces."""
    return re.sub(r"\s+", " ", value.replace("\u00a0", " ")).strip()


def parse_consolidation_reference(markup: str, expected_celex: str) -> tuple[str, date, str]:
    """Read the official consolidation header from the English XHTML expression."""
    match = _REFERENCE_RE.search(markup)
    if match is None:
        raise DocumentCollectionError(
            f"CELLAR XHTML for {expected_celex} has no English consolidation header."
        )
    base_celex = match.group("celex")
    version_date = date(
        int(match.group("year")),
        int(match.group("month")),
        int(match.group("day")),
    )
    consolidated_celex = f"{base_celex}-{version_date.strftime('%Y%m%d')}"
    if consolidated_celex != expected_celex:
        raise DocumentCollectionError(
            f"CELLAR XHTML header is {consolidated_celex}, expected {expected_celex}."
        )
    reference = (
        f"{base_celex} — EN — {match.group('day')}.{match.group('month')}."
        f"{match.group('year')} — {match.group('version')}"
    )
    return reference, version_date, match.group("version")


def parse_expression_title(rdf_xml: str, expected_celex: str) -> str:
    """Read the English expression title from CELLAR RDF."""
    match = _TITLE_RE.search(rdf_xml)
    if match is None:
        raise DocumentCollectionError(
            f"CELLAR RDF for {expected_celex} has no expression title."
        )
    title = normalize_whitespace(match.group(1))
    if not title:
        raise DocumentCollectionError(
            f"CELLAR RDF for {expected_celex} has an empty expression title."
        )
    return title


def parse_official_filename(listing_html: str, celex: str) -> str:
    """Read the single official PDF filename from a CELLAR content-stream list."""
    filenames = _FILENAME_RE.findall(listing_html)
    unique_names = list(dict.fromkeys(filenames))
    if len(unique_names) != 1:
        raise DocumentCollectionError(
            f"Expected one official PDF name for {celex}, found {unique_names}."
        )
    return unique_names[0]


def resolve_official_pdf(celex: str) -> ResolvedDocument:
    """Find the English PDF/A manifestation of a consolidated CELEX number."""
    rdf_xml = _fetch_text(
        f"{PUBLICATIONS_BASE}/celex/{celex}",
        accept="application/rdf+xml",
    )
    title = parse_expression_title(rdf_xml, celex)
    pdf_resource = _pdf_resource_url(rdf_xml, celex)
    download_url = _pdf_item_url(pdf_resource, celex)
    listing = _fetch_text(_list_url(download_url), accept="application/list;mtype=fmx4")
    filename = parse_official_filename(listing, celex)
    _require_filename_matches_celex(filename, celex)
    xhtml = _fetch_text(
        f"{PUBLICATIONS_BASE}/celex/{celex}",
        accept="application/xhtml+xml",
        maximum_bytes=65536,
    )
    reference, version_date, _version = parse_consolidation_reference(xhtml, celex)
    return ResolvedDocument(
        title=title,
        consolidation_reference=reference,
        version_date=version_date,
        consolidation_version=_version,
        official_filename=filename,
        download_url=download_url,
    )


def download_pdf(url: str, destination: object) -> None:
    """Stream an official PDF to destination without altering its bytes."""
    from pathlib import Path

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    request = _request(url, PDF_ACCEPT)
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            content_type = response.headers.get("Content-Type", "")
            if "pdf" not in content_type.lower():
                raise DocumentCollectionError(
                    f"Download URL did not return a PDF ({content_type or 'no content type'}): {url}"
                )
            with path.open("wb") as handle:
                while True:
                    chunk = response.read(256 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
    except HTTPError as exc:
        raise DocumentCollectionError(f"PDF download failed for {url}: HTTP {exc.code}") from exc
    except URLError as exc:
        raise DocumentCollectionError(f"PDF download failed for {url}: {exc.reason}") from exc


def _pdf_resource_url(rdf_xml: str, celex: str) -> str:
    match = _PDF_RESOURCE_RE.search(rdf_xml)
    if match is None:
        raise DocumentCollectionError(f"CELLAR RDF for {celex} has no PDF/A manifestation.")
    return _use_https(match.group(0))


def _pdf_item_url(pdf_resource_url: str, celex: str) -> str:
    request = _request(pdf_resource_url, PDF_ACCEPT)
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            final_url = response.geturl()
            content_type = response.headers.get("Content-Type", "")
    except HTTPError as exc:
        raise DocumentCollectionError(
            f"PDF manifestation lookup failed for {celex}: HTTP {exc.code}"
        ) from exc
    except URLError as exc:
        raise DocumentCollectionError(
            f"PDF manifestation lookup failed for {celex}: {exc.reason}"
        ) from exc
    if "pdf" not in content_type.lower() and not final_url.rstrip("/").endswith("/DOC_1"):
        raise DocumentCollectionError(
            f"PDF manifestation for {celex} resolved to {final_url} ({content_type})."
        )
    return _use_https(final_url)


def _list_url(download_url: str) -> str:
    if "/DOC_1" not in download_url:
        raise DocumentCollectionError(f"Unexpected PDF item URL: {download_url}")
    return download_url.split("/DOC_1", maxsplit=1)[0] + "/list"


def _require_filename_matches_celex(filename: str, celex: str) -> None:
    """Reject a stream whose Publications Office filename is for another act."""
    body = celex.split("-", maxsplit=1)[0]
    year = body[1:5]
    kind = body[5]
    number = body[6:]
    token = f"{year}{kind}{number}"
    if token not in filename:
        raise DocumentCollectionError(
            f"Official filename {filename} does not match CELEX {celex}."
        )


def _fetch_text(url: str, accept: str, maximum_bytes: int | None = None) -> str:
    request = _request(url, accept)
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            if maximum_bytes is None:
                payload = response.read()
            else:
                payload = response.read(maximum_bytes)
    except HTTPError as exc:
        raise DocumentCollectionError(f"CELLAR request failed for {url}: HTTP {exc.code}") from exc
    except URLError as exc:
        raise DocumentCollectionError(f"CELLAR request failed for {url}: {exc.reason}") from exc
    return payload.decode("utf-8", errors="replace")


def _request(url: str, accept: str) -> Request:
    return Request(
        _use_https(url),
        headers={
            "User-Agent": USER_AGENT,
            "Accept": accept,
            "Accept-Language": "eng",
            "Accept-Encoding": "identity",
        },
    )


def _use_https(url: str) -> str:
    if url.startswith("http://publications.europa.eu/"):
        return "https://" + url[len("http://") :]
    return url
