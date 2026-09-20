# -*- coding: utf-8 -*-
"""Triple decomposition of Vietnamese characters.

Every Vietnamese character is decomposed into three orthogonal components:

    base      : the underlying Latin letter (case-sensitive) or symbol
    modifier  : vowel-shape modifier  {none, circumflex(^), breve(˘), horn(+), stroke(đ)}
    tone      : tone mark             {level, acute(sắc), grave(huyền),
                                       hook(hỏi), tilde(ngã), dot(nặng)}

e.g.  'ệ' -> ('e', CIRCUMFLEX, DOT)      'ắ' -> ('a', BREVE, ACUTE)
      'đ' -> ('d', STROKE, LEVEL)        'M' -> ('M', NONE, LEVEL)

Rationale: a 229-class combined vocabulary is extremely sparse on small
scene-text datasets (VinText: 1200 train images). Decomposition reduces the
label space to ~95 bases x 5 modifiers x 6 tones and lets each prediction
head see dozens of times more samples per class.
"""
import unicodedata

# ---------------------------------------------------------------- constants
MODIFIERS = ("none", "circumflex", "breve", "horn", "stroke")
TONES = ("level", "acute", "grave", "hook", "tilde", "dot")
NUM_MODIFIERS = len(MODIFIERS)   # 5
NUM_TONES = len(TONES)           # 6

MOD_NONE, MOD_CIRCUMFLEX, MOD_BREVE, MOD_HORN, MOD_STROKE = range(5)
TONE_LEVEL, TONE_ACUTE, TONE_GRAVE, TONE_HOOK, TONE_TILDE, TONE_DOT = range(6)

_MOD_MARKS = {"̂": MOD_CIRCUMFLEX, "̆": MOD_BREVE, "̛": MOD_HORN}
_TONE_MARKS = {"́": TONE_ACUTE, "̀": TONE_GRAVE,
               "̉": TONE_HOOK, "̃": TONE_TILDE, "̣": TONE_DOT}
_MOD_TO_MARK = {v: k for k, v in _MOD_MARKS.items()}
_TONE_TO_MARK = {v: k for k, v in _TONE_MARKS.items()}

_VOWELS = set("aeiouy")
# which modifiers a (lowercased) base letter admits, besides MOD_NONE
_MOD_DOMAIN = {
    MOD_CIRCUMFLEX: set("aeo"),
    MOD_BREVE: set("a"),
    MOD_HORN: set("ou"),
    MOD_STROKE: set("d"),
}


# ---------------------------------------------------------------- functions
def decompose_char(ch):
    """Return (base, modifier_id, tone_id) for a single character.

    Characters outside the Vietnamese alphabet come back unchanged as
    (ch, MOD_NONE, TONE_LEVEL). If the character carries combining marks we
    do not model, it is treated as atomic.
    """
    if ch == "đ":                        # đ
        return "d", MOD_STROKE, TONE_LEVEL
    if ch == "Đ":                        # Đ
        return "D", MOD_STROKE, TONE_LEVEL
    nfd = unicodedata.normalize("NFD", ch)
    if len(nfd) == 1:
        return ch, MOD_NONE, TONE_LEVEL
    base, mod, tone = nfd[0], MOD_NONE, TONE_LEVEL
    for mark in nfd[1:]:
        if mark in _MOD_MARKS and mod == MOD_NONE:
            mod = _MOD_MARKS[mark]
        elif mark in _TONE_MARKS and tone == TONE_LEVEL:
            tone = _TONE_MARKS[mark]
        else:                                  # unknown / duplicate mark
            return ch, MOD_NONE, TONE_LEVEL
    if not is_valid_triple(base, mod, tone):
        return ch, MOD_NONE, TONE_LEVEL
    return base, mod, tone


def compose_char(base, mod=MOD_NONE, tone=TONE_LEVEL):
    """Inverse of decompose_char. Raises ValueError on invalid triples."""
    if not is_valid_triple(base, mod, tone):
        raise ValueError(f"invalid triple: ({base!r}, {mod}, {tone})")
    if mod == MOD_STROKE:
        return "Đ" if base == "D" else "đ"
    s = base
    if mod != MOD_NONE:
        s += _MOD_TO_MARK[mod]
    if tone != TONE_LEVEL:
        s += _TONE_TO_MARK[tone]
    return unicodedata.normalize("NFC", s)


def is_valid_triple(base, mod, tone):
    """Linguistic validity of a (base, modifier, tone) combination."""
    low = base.lower()
    if mod != MOD_NONE:
        if len(base) != 1 or low not in _MOD_DOMAIN[mod]:
            return False
        if mod == MOD_STROKE and tone != TONE_LEVEL:
            return False
    if tone != TONE_LEVEL and (len(base) != 1 or low not in _VOWELS):
        return False
    return True


def decompose_text(text):
    """Decompose a string into three parallel id/char sequences."""
    return [decompose_char(c) for c in unicodedata.normalize("NFC", text)]


def compose_text(triples):
    out = []
    for base, mod, tone in triples:
        try:
            out.append(compose_char(base, mod, tone))
        except ValueError:
            out.append(base)
    return "".join(out)


# ---------------------------------------------------------------- charset
class Charset:
    """Maps between full characters, triple ids and model label ids.

    Built from the recognition charset (e.g. VinText's 228 characters).
    Bases absent from the charset but reachable by stripping marks are added
    automatically, so every charset char has a valid (b, m, t) encoding.
    """

    def __init__(self, chars):
        chars = list(dict.fromkeys(chars))            # dedup, keep order
        bases = []
        for ch in chars:
            b, _, _ = decompose_char(ch)
            if b not in bases:
                bases.append(b)
        self.bases = bases
        self.base_stoi = {b: i for i, b in enumerate(bases)}
        self.chars = chars

        # full char -> (base_id, mod_id, tone_id)
        self.char_to_triple = {}
        for ch in chars:
            b, m, t = decompose_char(ch)
            self.char_to_triple[ch] = (self.base_stoi[b], m, t)

        # validity mask over (base, mod, tone); invalid triples are never
        # decoded and their probability mass is penalised during training
        self.valid = [[[is_valid_triple(b, m, t)
                        for t in range(NUM_TONES)]
                       for m in range(NUM_MODIFIERS)]
                      for b in bases]

    @classmethod
    def from_file(cls, path):
        with open(path, encoding="utf-8") as f:
            content = f.read()
        chars = [c for c in content if c not in "\r\n"]
        return cls(chars)

    @property
    def num_bases(self):
        return len(self.bases)

    def encode(self, text):
        """text -> list of (base_id, mod_id, tone_id); None for OOV chars."""
        out = []
        for ch in unicodedata.normalize("NFC", text):
            if ch in self.char_to_triple:
                out.append(self.char_to_triple[ch])
                continue
            b, m, t = decompose_char(ch)
            out.append((self.base_stoi[b], m, t) if b in self.base_stoi else None)
        return out

    def decode(self, triples):
        """list of (base_id, mod_id, tone_id) -> text (invalid -> base only)."""
        out = []
        for bi, m, t in triples:
            base = self.bases[bi]
            if not self.valid[bi][m][t]:
                m, t = MOD_NONE, TONE_LEVEL
            out.append(compose_char(base, m, t))
        return "".join(out)
