"""Stage 1 of the English reading bank: cut public-domain books into 35-60 word passages.

Passages start at a paragraph start (so they rarely begin mid-thought), stop at a sentence
end, and are filtered for things that make a bad reading exercise: digits, headings, rare or
archaic words, too many names, footnote marks. Output: data/bank/candidates.jsonl.

    .venv/bin/python tools/bank_candidates.py
"""
import json
import random
import re
import sys
from pathlib import Path

from wordfreq import zipf_frequency

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"
OUT = ROOT / "data" / "bank" / "candidates.jsonl"

BOOKS = {  # Gutenberg id: (author, title)
    308: ("Jerome K. Jerome", "Three Men in a Boat"),
    849: ("Jerome K. Jerome", "Idle Thoughts of an Idle Fellow"),
    269: ("Saki", "Beasts and Super-Beasts"),
    3688: ("Saki", "The Chronicles of Clovis"),
    1661: ("Arthur Conan Doyle", "The Adventures of Sherlock Holmes"),
    834: ("Arthur Conan Doyle", "The Memoirs of Sherlock Holmes"),
    21: ("Aesop", "Fables"),
    2776: ("O. Henry", "The Four Million"),
    6340: ("Stephen Leacock", "Literary Lapses"),
    8164: ("P. G. Wodehouse", "My Man Jeeves"),
    10554: ("P. G. Wodehouse", "Right Ho, Jeeves"),
    8092: ("G. K. Chesterton", "Tremendous Trifles"),
    1952: ("Charlotte Perkins Gilman", "The Yellow Wallpaper"),
    2148: ("Edgar Allan Poe", "Works, Volume II"),
    3176: ("Mark Twain", "The Innocents Abroad"),
}

MIN_WORDS, MAX_WORDS = 35, 60
ARCHAIC = {"thee", "thou", "thy", "thine", "hath", "doth", "art", "ye", "shalt", "wilt", "hast",
           "dost", "nay", "'tis", "twas", "methinks", "unto", "whence", "thither", "hither"}
WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")
SENT_END = re.compile(r'(?<!\bMr\.)(?<!\bMrs\.)(?<!\bDr\.)(?<!\bSt\.)(?:(?<=[.!?])|(?<=[.!?]["”’]))\s+(?=["“‘]?[A-Z])')


def body(text):
    a = text.find("*** START OF")
    b = text.find("*** END OF")
    text = text[text.find("\n", a) + 1 if a >= 0 else 0: b if b >= 0 else len(text)]
    return text.replace("\r", "")


def clean(par):
    par = re.sub(r"\[[^\]]*\]", "", par)  # [Illustration], [Footnote 3]
    par = re.sub(r"_([^_]+)_", r"\1", par)  # _italics_
    par = par.replace("--", "—").replace("``", "“").replace("''", "”")
    par = re.sub(r"\s+", " ", par).strip()
    return par


def paragraphs(text):
    for block in re.split(r"\n\s*\n", body(text)):
        lines = block.strip().split("\n")
        if not lines or not lines[0].strip():
            continue
        p = clean(" ".join(lines))
        # headings and titles: short and not ending like a sentence
        if p and (len(p.split()) > 12 or re.search(r'[.!?:;]["”’)]?$', p)):
            yield p


def bad_paragraph(p):
    letters = [c for c in p if c.isalpha()]
    if len(letters) < 20:
        return True
    if sum(c.isupper() for c in letters) > 0.3 * len(letters):  # headings, CONTENTS
        return True
    return bool(re.search(r"\d|Gutenberg|CHAPTER|Chapter [IVXL]+|\*", p))


def word_problems(text):
    words = WORD.findall(text)
    rare = names = 0
    for w in words:
        lw = w.lower().replace("’", "'")
        if lw in ARCHAIC:
            return len(words), 99, 99
        z = zipf_frequency(lw, "en")
        if w[0].isupper() and z < 4.0 and lw not in ("i",):
            names += 1
        elif z < 2.5:
            rare += 1
    return len(words), rare, names


def passages(pars):
    """Greedy: from each paragraph start, add sentences (crossing into following paragraphs)
    until MIN_WORDS..MAX_WORDS is reached at a sentence end."""
    for i in range(len(pars)):  # noqa
        sents, n = [], 0
        for p in pars[i:i + 6]:
            for s in SENT_END.split(p):
                sents.append(s)
                n += len(WORD.findall(s))
                if n > MAX_WORDS:
                    break
                if n >= MIN_WORDS and re.search(r'[.!?]["”’]?$', s):
                    yield " ".join(sents)
                    break
            else:
                continue
            break


def main():
    out = []
    for gid, (author, title) in BOOKS.items():
        path = CORPUS / f"pg{gid}.txt"
        if not path.exists():
            print("missing", path, file=sys.stderr)
            continue
        pars = [p for p in paragraphs(path.read_text(encoding="utf-8", errors="replace"))]
        kept = 0
        for text in passages(pars):
            if bad_paragraph(text):
                continue
            n, rare, names = word_problems(text)
            if rare > 1 or names > 3:
                continue
            if text.count("“") + text.count('"') > 6:  # rapid-fire dialogue lines
                continue
            out.append(dict(text=text, author=author, title=title, gid=gid, words=n, rare=rare,
                            names=names))
            kept += 1
        print(f"{title:40s} paragraphs {len(pars):5d}  passages kept {kept:5d}")
    # drop passages that overlap heavily (consecutive starts often share sentences)
    random.seed(0)
    uniq, used = [], set()
    for c in out:
        key = set(re.findall(r"[^.!?]+[.!?]", c["text"]))
        if key & used:
            continue
        used |= key
        uniq.append(c)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for c in uniq:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"{len(out)} passages, {len(uniq)} without overlap -> {OUT}")


if __name__ == "__main__":
    main()
