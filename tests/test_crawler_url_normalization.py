import pytest

import crawler


def test_canonicalize_url_strips_query_fragment_and_trailing_slash():
    u = crawler.canonicalize_url("https://www.appdome.com/how-to/?a=1#frag")
    assert u == "https://www.appdome.com/how-to"


def test_allowed_prefix_matches_seed_after_canonicalization():
    # Regression test for the "Visited=0" crawl bug caused by a trailing-slash mismatch.
    seed = crawler.canonicalize_url(crawler.START_URL)

    # ALLOWED_PREFIX should be compatible with canonicalized URLs.
    assert seed == crawler.ALLOWED_PREFIX or seed.startswith(crawler.ALLOWED_PREFIX + "/")
