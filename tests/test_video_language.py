from ingestion.extractors import video_transcriber as vt


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _mediator_payload(vtt_url=None, mp4_url="https://cdn/x.mp4"):
    files = [{"progressiveDownloadURL": mp4_url, "filesize": 10}]
    if vtt_url:
        files[0]["subtitles"] = {"url": vtt_url}
    return {"media": [{"title": "T", "duration": 60, "files": files}]}


def test_twi_uses_registry_mediator_code(monkeypatch):
    calls = []

    def fake_get(url, **_):
        calls.append(url)
        return _FakeResponse(_mediator_payload())

    monkeypatch.setattr("httpx.get", fake_get)
    result = vt._resolve_jworg_url(
        "https://www.jw.org/tw/nhomakorabea/videos/#tw/mediaitems/VODBible/pub-a_VIDEO"
    )
    assert "/media-items/TW/pub-a_VIDEO" in calls[0]
    assert result["language"] == "tw"
    assert result["whisper"] is False


def test_english_mediator_code_unchanged(monkeypatch):
    calls = []
    monkeypatch.setattr("httpx.get", lambda url, **_: calls.append(url) or _FakeResponse(_mediator_payload()))
    result = vt._resolve_jworg_url(
        "https://www.jw.org/en/library/videos/#en/mediaitems/LatestVideos/pub-b_VIDEO"
    )
    assert "/media-items/E/pub-b_VIDEO" in calls[0]
    assert result["language"] == "en"
    assert result["whisper"] is True


def test_twi_without_vtt_skips_whisper(monkeypatch):
    monkeypatch.setattr(
        vt,
        "_resolve_jworg_url",
        lambda url: {
            "title": "T", "duration": 60.0, "vtt_url": None,
            "mp4_url": "https://cdn/x.mp4", "language": "tw", "whisper": False,
        },
    )

    def fail(*_, **__):
        raise AssertionError("Whisper path must not run for Twi")

    monkeypatch.setattr(vt, "_download_direct", fail)
    monkeypatch.setattr(vt, "transcribe_file", fail)
    assert vt.transcribe_url("https://www.jw.org/tw/x/#tw/mediaitems/C/pub-a_VIDEO") is None


def test_twi_vtt_transcript_tagged_with_language(monkeypatch):
    monkeypatch.setattr(
        vt,
        "_resolve_jworg_url",
        lambda url: {
            "title": "T", "duration": 60.0, "vtt_url": "https://cdn/x.vtt",
            "mp4_url": None, "language": "tw", "whisper": False,
        },
    )
    monkeypatch.setattr(vt, "_fetch_vtt_transcript", lambda url: "Onyankopɔn dɔ wiase")
    result = vt.transcribe_url("https://www.jw.org/tw/x/#tw/mediaitems/C/pub-a_VIDEO")
    assert result.language == "tw"
    assert result.metadata["language"] == "tw"


def test_forced_twi_skips_whisper_for_other_urls(monkeypatch):
    def fail(*_, **__):
        raise AssertionError("Whisper path must not run for a video forced to Twi")

    monkeypatch.setattr(vt, "_download_direct", fail)
    monkeypatch.setattr(vt, "_download_via_ytdlp", fail)
    assert vt.transcribe_url("https://youtube.com/watch?v=x", language="tw") is None


def test_forced_twi_skips_whisper_for_local_files(tmp_path):
    video = tmp_path / "talk.mp4"
    video.write_bytes(b"")
    assert vt.transcribe_file(video, language="tw") is None
