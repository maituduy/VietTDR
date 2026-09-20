# -*- coding: utf-8 -*-
import sys, os, unicodedata
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from viettdr.vietchar import (
    Charset, decompose_char, compose_char, decompose_text, compose_text,
    MOD_NONE, MOD_STROKE, TONE_LEVEL, NUM_MODIFIERS, NUM_TONES,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.path.join(_HERE, "..", "charset_vintext.txt"),
    os.path.join(_HERE, "..", "..", "DeepSolo", "DeepSolo-main", "DeepSolo",
                 "datasets", "vintext", "all_characters.txt"),
]
VINTEXT_CHARS = next((p for p in _CANDIDATES if os.path.exists(p)),
                     _CANDIDATES[0])


def test_roundtrip_vintext_vocab():
    cs = Charset.from_file(VINTEXT_CHARS)
    bad = []
    for ch in cs.chars:
        b, m, t = decompose_char(ch)
        try:
            back = compose_char(b, m, t)
        except ValueError:
            back = None
        if back != unicodedata.normalize("NFC", ch):
            bad.append((ch, b, m, t, back))
    assert not bad, f"roundtrip failed for: {bad}"
    print(f"[ok] roundtrip on {len(cs.chars)} VinText chars; "
          f"{cs.num_bases} bases x {NUM_MODIFIERS} modifiers x {NUM_TONES} tones")


def test_examples():
    assert decompose_char("ệ") == ("e", 1, 5)      # circumflex + dot
    assert decompose_char("ắ") == ("a", 2, 1)      # breve + acute
    assert decompose_char("ữ") == ("u", 3, 4)      # horn + tilde
    assert decompose_char("đ") == ("d", MOD_STROKE, TONE_LEVEL)
    assert decompose_char("M") == ("M", MOD_NONE, TONE_LEVEL)
    assert decompose_char("5") == ("5", MOD_NONE, TONE_LEVEL)
    assert compose_char("o", 3, 5) == "ợ"
    assert compose_text(decompose_text("CHẤT LƯỢNG TỐT Đẹp 123!")) == \
        "CHẤT LƯỢNG TỐT Đẹp 123!"
    print("[ok] examples")


def test_charset_encode_decode():
    cs = Charset.from_file(VINTEXT_CHARS)
    for word in ["NGUYỄN", "Kính", "mời", "trường", "ĐẠI"]:
        enc = cs.encode(word)
        assert None not in enc, word
        assert cs.decode(enc) == word, word
    # invalid triples are sanitised, not crashed on
    qi = cs.base_stoi["q"]
    assert cs.decode([(qi, 0, 3)]) == "q"
    print("[ok] charset encode/decode")


def test_validity_counts():
    cs = Charset.from_file(VINTEXT_CHARS)
    n_valid = sum(cs.valid[b][m][t]
                  for b in range(cs.num_bases)
                  for m in range(NUM_MODIFIERS)
                  for t in range(NUM_TONES))
    total = cs.num_bases * NUM_MODIFIERS * NUM_TONES
    # every charset char must map to a valid triple
    for ch, (b, m, t) in cs.char_to_triple.items():
        assert cs.valid[b][m][t], ch
    print(f"[ok] {n_valid}/{total} triples valid "
          f"({100.0 * n_valid / total:.1f}% of the product space)")


if __name__ == "__main__":
    test_roundtrip_vintext_vocab()
    test_examples()
    test_charset_encode_decode()
    test_validity_counts()
    print("ALL TESTS PASSED")
