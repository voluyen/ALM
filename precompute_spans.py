"""
Offline span extraction for MTA loss.

Reads a jsonl with `text` field, runs spaCy to extract noun-chunks + verb-phrases,
filters overlaps, and writes a new jsonl with two extra fields:
    spans_char_offsets : list[[start, end]]  -- multi-token spans
    words_char_offsets : list[[start, end]]  -- per-word offsets inside spans

Usage:
    python precompute_spans.py \
        --input  data/dolly_train.jsonl \
        --output data/dolly_train_with_spans.jsonl
"""

import argparse
import json
import os
import sys

import spacy
from spacy.matcher import Matcher
from tqdm import tqdm


VERB_PHRASE_PATTERN = [
    {"POS": "AUX",  "OP": "*"},
    {"POS": "ADV",  "OP": "*"},
    {"POS": "VERB", "OP": "+"},
    {"POS": "ADV",  "OP": "*"},
]


def filter_overlapping_spans(spans_with_docrefs):
    """
    Greedy non-overlap filter on (start, end, spacy_span) tuples.
    Keep the longest span when several share a starting char.

    Returns:
        filtered_spans : list[(start_char, end_char)]
        word_offsets   : list[(start_char, end_char)]  -- one per token inside each kept span
    """
    sorted_spans = sorted(spans_with_docrefs, key=lambda s: (s[0], -s[1]))
    filtered, words = [], []
    if not sorted_spans:
        return filtered, words

    current = sorted_spans[0]
    for nxt in sorted_spans[1:]:
        # Skip if nxt entirely covered by current
        if nxt[1] <= current[1]:
            continue
        filtered.append((current[0], current[1]))
        p = current[2]
        n = len(p)
        # word offsets: from each token start to the next token start (or token end for last)
        words.extend([(p[i - 1].idx, p[i].idx) for i in range(1, n)])
        words.append((p[n - 1].idx, p[n - 1].idx + len(p[n - 1])))
        current = nxt

    filtered.append((current[0], current[1]))
    p = current[2]
    n = len(p)
    words.extend([(p[i - 1].idx, p[i].idx) for i in range(1, n)])
    words.append((p[n - 1].idx, p[n - 1].idx + len(p[n - 1])))
    return filtered, words


def extract_spans_for_doc(doc, matcher):
    """Run noun-chunk + verb-phrase extraction on a single spaCy Doc."""
    spans_with_offsets = []
    # Verb phrases via Matcher
    for _, start, end in matcher(doc):
        vp = doc[start:end]
        spans_with_offsets.append((vp.start_char, vp.end_char, vp))
    # Noun chunks
    spans_with_offsets.extend(
        [(nc.start_char, nc.end_char, nc) for nc in doc.noun_chunks]
    )
    return filter_overlapping_spans(spans_with_offsets)


def load_nlp(model_name):
    """Load spaCy model; raise with a helpful message if missing."""
    try:
        nlp = spacy.load(model_name, disable=["ner", "lemmatizer"])
    except OSError as e:
        sys.stderr.write(
            f"spaCy model '{model_name}' not found. Install:\n"
            f"  python -m spacy download {model_name}\n"
        )
        raise e
    # Allow long instruction-style texts
    nlp.max_length = 2_000_000
    return nlp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  required=True, help="Input jsonl path")
    ap.add_argument("--output", required=True, help="Output jsonl path")
    ap.add_argument("--text-field", default="text",
                    help="Name of the text field in each row (default: text)")
    ap.add_argument("--spacy-model", default="en_core_web_sm")
    ap.add_argument("--n-process", type=int, default=1,
                    help="spaCy n_process (default 1 -- Windows-safe)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite output file if it exists")
    args = ap.parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        sys.stderr.write(
            f"Output {args.output} already exists. Pass --overwrite to replace.\n"
        )
        sys.exit(1)

    nlp = load_nlp(args.spacy_model)
    matcher = Matcher(nlp.vocab)
    matcher.add("VERB_PHRASE", [VERB_PHRASE_PATTERN])

    # Load all rows so we can pair docs with their original json safely
    with open(args.input, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    texts = [row[args.text_field] for row in rows]

    n_total = len(rows)
    n_written = 0
    n_empty_spans = 0

    pipe_kwargs = {"batch_size": args.batch_size}
    if args.n_process > 1:
        pipe_kwargs["n_process"] = args.n_process

    with open(args.output, "w", encoding="utf-8") as out_f:
        for row, doc in tqdm(
            zip(rows, nlp.pipe(texts, **pipe_kwargs)),
            total=n_total, desc="extract spans",
        ):
            spans, words = extract_spans_for_doc(doc, matcher)
            if len(spans) == 0:
                n_empty_spans += 1
            row["spans_char_offsets"] = [[s, e] for s, e in spans]
            row["words_char_offsets"] = [[s, e] for s, e in words]
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1

    print(
        f"Done. Wrote {n_written}/{n_total} rows to {args.output}. "
        f"Rows with empty spans: {n_empty_spans}."
    )


if __name__ == "__main__":
    main()
