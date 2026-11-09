import pytest

from tools import court_browser


def test_court_urls_are_official_homepages_only():
    for key in court_browser.COURTS:
        assert court_browser.court_url(key).startswith("https://")
        assert court_browser.court_url(key).endswith(".sudrf.ru/")


def test_unknown_court_is_rejected():
    with pytest.raises(KeyError):
        court_browser.court_url("unknown")
