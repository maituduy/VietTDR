# -*- coding: utf-8 -*-
"""Download open-licence fonts that cover the full Vietnamese alphabet.

Vietnamese needs far more than Latin-1: stacked marks such as ễ ộ ữ ẳ are
absent from many otherwise-good fonts, and a font that silently renders a
missing glyph as a box would poison the synthetic training set. Every file
is therefore verified against a checklist of the hardest characters and
discarded if any is missing.

    python tools/fetch_fonts.py --out fonts
"""
import argparse
import os
import re
import urllib.parse
import urllib.request

# Characters that only fonts with real Vietnamese support carry.
HARD = "ăâđêôơưĂÂĐÊÔƠƯ" \
       "ắằẳẵặấầẩẫậếềểễệốồổỗộớờởỡợứừửữựỳỵỷỹ" \
       "ẮẰẲẴẶẤẦẨẪẬẾỀỂỄỆỐỒỔỖỘỚỜỞỠỢỨỪỬỮỰỲỴỶỸ"

BASE = "https://github.com/google/fonts/raw/main/"
FONTS = [
    # Vietnamese-designed families first
    "ofl/bevietnampro/BeVietnamPro-Regular.ttf",
    "ofl/bevietnampro/BeVietnamPro-Bold.ttf",
    "ofl/bevietnampro/BeVietnamPro-Italic.ttf",
    # common UI / signage sans
    "ofl/opensans/OpenSans[wdth,wght].ttf",
    "ofl/notosans/NotoSans[wdth,wght].ttf",
    "ofl/montserrat/Montserrat[wght].ttf",
    "ofl/oswald/Oswald[wght].ttf",
    "ofl/nunito/Nunito[wght].ttf",
    "ofl/mulish/Mulish[wght].ttf",
    "ofl/quicksand/Quicksand[wght].ttf",
    "ofl/lexend/Lexend[wght].ttf",
    "ofl/inter/Inter[opsz,wght].ttf",
    # serif / display / handwriting — shop signs use plenty of these
    "ofl/lora/Lora[wght].ttf",
    "ofl/playfairdisplay/PlayfairDisplay[wght].ttf",
    "ofl/notoserif/NotoSerif[wdth,wght].ttf",
    "ofl/dancingscript/DancingScript[wght].ttf",
    "ofl/pacifico/Pacifico-Regular.ttf",
    "ofl/bungee/Bungee-Regular.ttf",
    "ofl/pattaya/Pattaya-Regular.ttf",
    "ofl/charm/Charm-Regular.ttf",
    "ofl/charmonman/Charmonman-Regular.ttf",
]


def covers_vietnamese(path):
    try:
        from fontTools.ttLib import TTFont
        cmap = TTFont(path, fontNumber=0, lazy=True).getBestCmap()
    except Exception as e:                      # noqa: BLE001
        return False, f"unreadable ({e.__class__.__name__})"
    missing = [c for c in HARD if ord(c) not in cmap]
    return (not missing), (f"missing {len(missing)}: {''.join(missing[:8])}"
                           if missing else "ok")


def safe_name(name):
    """Kaggle rejects dataset files whose names contain '[' or ']', and that
    is exactly how Google ships variable fonts (Montserrat[wght].ttf). Store
    them as Montserrat-wght.ttf so the bundle can be uploaded as-is."""
    stem, ext = os.path.splitext(name)
    stem = re.sub(r"\[([^\]]*)\]", r"-\1", stem)
    stem = re.sub(r"[^A-Za-z0-9._-]", "-", stem)
    stem = re.sub(r"-{2,}", "-", stem).strip("-")
    return stem + ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fonts")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    kept = 0
    for rel in FONTS:
        name = safe_name(os.path.basename(rel))
        dst = os.path.join(args.out, name)
        if not os.path.exists(dst):
            # variable-font filenames contain [] which must be percent-encoded
            d, f = rel.rsplit("/", 1)
            url = BASE + d + "/" + urllib.parse.quote(f)
            try:
                urllib.request.urlretrieve(url, dst)
            except Exception as e:              # noqa: BLE001
                print(f"  [skip] {name}: download failed ({e})")
                continue
        ok, why = covers_vietnamese(dst)
        if ok:
            kept += 1
            print(f"  [keep] {name}")
        else:
            os.remove(dst)
            print(f"  [drop] {name}: {why}")
    print(f"\n{kept} usable fonts in {args.out}/")
    if kept < 5:
        print("WARNING: too few fonts - synthetic data will lack diversity")


if __name__ == "__main__":
    main()
