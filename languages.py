"""
Language registry
─────────────────
Single source of truth for every per-language fact the ingestion pipeline and
the chat API need: jw.org locale and mediator code, Pinecone namespace, Bible
publication, excluded paths, Bible book names and canned replies.

Adding a language means adding one ``Language`` entry here and listing its
code in ``ENABLED_LANGUAGES``; no other module hardcodes a language.

English keeps the legacy default namespace ("") so existing vectors are never
rewritten.
"""

import os
from dataclasses import dataclass, field
from urllib.parse import urlparse


@dataclass(frozen=True)
class Messages:
    greeting: str
    out_of_domain: str
    unavailable: str
    no_results: str
    no_text: str
    not_found: str
    english_sources_note: str


@dataclass(frozen=True)
class Language:
    code: str                         # ISO 639-1, also our id: "en", "tw"
    name: str                         # English name, used in prompts: "Twi"
    native_name: str                  # shown in the UI picker
    jw_locale: str                    # jw.org path segment: "tw"
    jw_code: str                      # mediator / finder wtlocale: "TW"
    namespace: str                    # Pinecone namespace: "" for en (legacy)
    bible_pub: str                    # finder pub: "nwtsty" / "nwt"
    excluded_paths: tuple[str, ...]   # path substrings never ingested
    whisper: bool                     # Whisper can transcribe this language
    min_score: float | None           # overrides RETRIEVAL_MIN_SCORE when set
    book_names: tuple[str, ...]       # 66 NWT book names, index = book number - 1
    messages: Messages
    bm25_fold: dict[str, str] = field(default_factory=dict)  # folded before BM25 tokenising
    aliases: tuple[str, ...] = ()     # other codes a detector may return ("ak", "twi")


# Scraped from https://www.jw.org/en/library/bible/study-bible/books/
_EN_BOOKS = (
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua", "Judges",
    "Ruth", "1 Samuel", "2 Samuel", "1 Kings", "2 Kings", "1 Chronicles",
    "2 Chronicles", "Ezra", "Nehemiah", "Esther", "Job", "Psalms", "Proverbs",
    "Ecclesiastes", "Song of Solomon", "Isaiah", "Jeremiah", "Lamentations",
    "Ezekiel", "Daniel", "Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah",
    "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi", "Matthew",
    "Mark", "Luke", "John", "Acts", "Romans", "1 Corinthians", "2 Corinthians",
    "Galatians", "Ephesians", "Philippians", "Colossians", "1 Thessalonians",
    "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews",
    "James", "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude", "Revelation",
)

# Scraped from https://www.jw.org/tw/nhomakorabea/bible/nwt/nhoma/
_TW_BOOKS = (
    "Genesis", "Exodus", "Leviticus", "Numeri", "Deuteronomium", "Yosua", "Atemmufo",
    "Rut", "1 Samuel", "2 Samuel", "1 Ahene", "2 Ahene", "1 Beresosɛm", "2 Beresosɛm",
    "Esra", "Nehemia", "Ester", "Hiob", "Nnwom", "Mmebusɛm", "Ɔsɛnkafo", "Solomon Dwom",
    "Yesaia", "Yeremia", "Kwadwom", "Hesekiel", "Daniel", "Hosea", "Yoel", "Amos",
    "Obadia", "Yona", "Mika", "Nahum", "Habakuk", "Sefania", "Hagai", "Sakaria",
    "Malaki", "Mateo", "Marko", "Luka", "Yohane", "Asomafo Nnwuma", "Romafo",
    "1 Korintofo", "2 Korintofo", "Galatifo", "Efesofo", "Filipifo", "Kolosefo",
    "1 Tesalonikafo", "2 Tesalonikafo", "1 Timoteo", "2 Timoteo", "Tito", "Filemon",
    "Hebrifo", "Yakobo", "1 Petro", "2 Petro", "1 Yohane", "2 Yohane", "3 Yohane",
    "Yuda", "Adiyisɛm",
)

ENGLISH = Language(
    code="en",
    name="English",
    native_name="English",
    jw_locale="en",
    jw_code="E",
    namespace="",
    bible_pub="nwtsty",
    excluded_paths=("/library/bible/",),
    whisper=True,
    min_score=None,
    book_names=_EN_BOOKS,
    messages=Messages(
        greeting=(
            "Hello. I can help with questions based on indexed Jehovah's Witnesses content. "
            "Ask me a question about JW teachings, publications, or material from jw.org."
        ),
        out_of_domain=(
            "I can only answer from indexed Jehovah's Witnesses sources. "
            "Please ask a JW-related question so I can cite the relevant material."
        ),
        unavailable=(
            "The AI model is currently unavailable, so I can't process your question right "
            "now. Please try again in a moment."
        ),
        no_results="I could not find relevant content in the indexed books yet.",
        no_text="I found matches, but they did not contain readable text content.",
        not_found="I could not find information about this in the provided sources.",
        english_sources_note="",
    ),
)

# DRAFT: the Twi messages below are machine-drafted and must be reviewed by a
# fluent Twi speaker before Twi is enabled in production.
TWI = Language(
    code="tw",
    name="Twi",
    native_name="Twi",
    jw_locale="tw",
    jw_code="TW",
    namespace="tw",
    bible_pub="nwt",
    excluded_paths=("/nhomakorabea/bible/",),
    whisper=False,  # Whisper has no Twi/Akan model; only VTT subtitles are used
    min_score=None,  # set from the PR 3 eval's score percentiles
    book_names=_TW_BOOKS,
    messages=Messages(
        greeting=(
            "Akwaaba. Metumi aboa wo wɔ nsɛmmisa a ɛfa Yehowa Adansefo nhoma ne jw.org so "
            "nsɛm ho. Bisa me asɛm biara a ɛfa wɔn nkyerɛkyerɛ anaa wɔn nhoma ho."
        ),
        out_of_domain=(
            "Metumi ama mmuae afi Yehowa Adansefo nhoma a ɛwɔ me nkyɛn no mu nko ara. "
            "Yɛsrɛ wo, bisa asɛm a ɛfa Yehowa Adansefo ho na makyerɛ baabi a mmuae no fi."
        ),
        unavailable=(
            "AI no nni hɔ seesei, enti mintumi nnyi w'asɛmmisa no ano mprempren. "
            "Yɛsrɛ wo, san bɔ mmɔden bere tiaa bi akyi."
        ),
        no_results="Minhuu nsɛm a ɛfa ho wɔ nhoma a ɛwɔ me nkyɛn no mu.",
        no_text="Mehuu nsɛm bi, nanso na nkyerɛwee biara nni mu a wotumi kenkan.",
        not_found="Minhuu nsɛm a ɛfa eyi ho wɔ nhoma a wɔde ama me no mu.",
        english_sources_note=(
            "Minhuu Twi nhoma pii a ɛfa eyi ho, enti mmuae yi gyina Borɔfo nhoma so."
        ),
    ),
    bm25_fold={"ɛ": "e", "ɔ": "o", "Ɛ": "E", "Ɔ": "O"},
    aliases=("ak", "aka", "twi", "akan"),
)

LANGUAGES: dict[str, Language] = {lang.code: lang for lang in (ENGLISH, TWI)}
DEFAULT = ENGLISH


def get(code: str) -> Language:
    """Return the registered language for ``code``; raises KeyError if unknown."""
    return LANGUAGES[code.strip().lower()]


def normalise_code(raw: str | None) -> str:
    """Map a detector's language label to a registry code where possible
    ("ak", "Twi" -> "tw", "en-US" -> "en"). Unregistered codes are returned
    lowercased, and an empty label becomes ""."""
    code = (raw or "").strip().lower().replace("_", "-").split("-")[0]
    if code in LANGUAGES:
        return code
    for lang in LANGUAGES.values():
        if code in lang.aliases or code == lang.name.lower():
            return lang.code
    return code


def bm25_fold_table() -> dict[int, str]:
    """str.translate table folding every language's special letters, so
    "Onyankopon" and "Onyankopɔn" tokenise the same for BM25."""
    folds: dict[str, str] = {}
    for lang in LANGUAGES.values():
        folds.update(lang.bm25_fold)
    return str.maketrans(folds)


def enabled() -> list[Language]:
    """Languages listed in ENABLED_LANGUAGES (default "en"). English is always
    included because it is the retrieval fallback for every other language."""
    raw = os.getenv("ENABLED_LANGUAGES", DEFAULT.code)
    codes = [c.strip().lower() for c in raw.split(",") if c.strip()]
    result = [DEFAULT]
    for code in codes:
        lang = LANGUAGES.get(code)
        if lang and lang not in result:
            result.append(lang)
    return result


def excluded_paths() -> tuple[str, ...]:
    """Union of every registered language's excluded path substrings."""
    paths: list[str] = []
    for lang in LANGUAGES.values():
        for path in lang.excluded_paths:
            if path not in paths:
                paths.append(path)
    return tuple(paths)


def from_jw_locale(locale: str) -> Language | None:
    locale = (locale or "").strip().lower()
    return next((lang for lang in LANGUAGES.values() if lang.jw_locale == locale), None)


def from_jw_url(url: str) -> Language | None:
    """Language of a jw.org URL from its first path segment (``/tw/...``), or None
    for non-jw.org URLs and unregistered locales."""
    parsed = urlparse(url or "")
    host = (parsed.hostname or "").lower()
    if host != "jw.org" and not host.endswith(".jw.org"):
        return None
    segments = [s for s in parsed.path.split("/") if s]
    return from_jw_locale(segments[0]) if segments else None


def resolve_ingest_language(url: str, override: str | None = None) -> Language:
    """Language to tag an ingested document with: an explicit override wins,
    then the jw.org URL locale, then English."""
    if override:
        return get(override)
    return from_jw_url(url) or DEFAULT
