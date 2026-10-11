from engine.vocab import parse_class


def test_parse_direct():
    assert parse_class("影像中的主要地物是山脉。") == "mountain"


def test_parse_synonym():
    assert parse_class("这是飞机场") == "airport"


def test_parse_none():
    assert parse_class("看不太出来") is None


def test_parse_bridge_substring():
    assert parse_class("画面里有一座桥") == "bridge"


def test_parse_prefers_longer_canonical():
    # 立交桥 是标准类名，不应被 "桥" 同义词截胡
    assert parse_class("图中是立交桥") == "overpass"
