import argparse
import csv
import json
import sqlite3
from pathlib import Path

TEMPLATES = {'plain': '{question}', 'label': '{label}: {question}'}


def windows(text: str, size: int, stride: int) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [' '.join(words)] if words else []
    out = []
    for start in range(0, len(words), stride):
        out.append(' '.join(words[start:start + size]))
        if start + size >= len(words):
            break
    return out


def load_questions(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    questions = {}
    for line in open(path):
        if not line.strip():
            continue
        topic = json.loads(line)
        for question in topic['questions']:
            questions[(topic['tid'], question['qid'])] = {
                'question': question['question'],
                'label': topic['label'].replace('-', ' ').replace('_', ' '),
                'narrative': topic['narrative'],
            }
    return questions


def main():
    parser = argparse.ArgumentParser(description='Score labelled (question, document) pairs with a cross-encoder to see whether its score separates relevant from irrelevant candidates')
    parser.add_argument('--items', type=Path, default=Path('data/rerank_test_items.csv'))
    parser.add_argument('--topics', type=Path, default=Path('data/dev-topics-all-docs.jsonl'))
    parser.add_argument('--db', type=Path, default=Path('docs.db'))
    parser.add_argument('--model', default='cross-encoder/ms-marco-MiniLM-L-6-v2')
    parser.add_argument('--device', default=None)
    parser.add_argument('--window', type=int, default=100, help='words per passage')
    parser.add_argument('--stride', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--out', type=Path, help='defaults to runs/rerank_test_<model>.csv')
    parser.add_argument('--dry-run', action='store_true', help='build the pairs and print a summary without loading the model')
    args = parser.parse_args()

    questions = load_questions(args.topics)
    rows = list(csv.DictReader(open(args.items)))
    db = sqlite3.connect(str(args.db))

    pairs: list[tuple[str, str]] = []
    owners: list[tuple[int, str, int]] = []
    passages_per_row = []
    for i, row in enumerate(rows):
        found = db.execute('select text from documents where id=?', (row['doc_id'],)).fetchone()
        passages = windows(found[0], args.window, args.stride) if found else []
        passages_per_row.append(len(passages))
        fields = questions[(row['tid'], row['qid'])]
        for name, template in TEMPLATES.items():
            query = template.format(**fields)
            for j, passage in enumerate(passages):
                pairs.append((query, passage))
                owners.append((i, name, j))

    print(f'{len(rows)} items, {len(pairs)} pairs, {sum(1 for n in passages_per_row if n == 0)} items with no text')
    if args.dry_run:
        for query, passage in pairs[:2]:
            print(f'  query: {query}\n  passage: {passage[:140]}')
        return

    from sentence_transformers import CrossEncoder

    model = CrossEncoder(args.model, device=args.device)
    scores = model.predict(pairs, batch_size=args.batch_size, show_progress_bar=True)

    best: dict[tuple[int, str], float] = {}
    lead: dict[tuple[int, str], float] = {}
    for (i, name, j), score in zip(owners, scores):
        score = float(score)
        if (i, name) not in best or score > best[(i, name)]:
            best[(i, name)] = score
        if j == 0:
            lead[(i, name)] = score

    out = args.out or Path('runs') / f'rerank_test_{args.model.replace("/", "__")}.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([*rows[0].keys(), 'passages', 'ce_max_plain', 'ce_lead_plain', 'ce_max_label', 'ce_lead_label', 'model'])
        for i, row in enumerate(rows):
            writer.writerow([*row.values(), passages_per_row[i],
                             best.get((i, 'plain')), lead.get((i, 'plain')),
                             best.get((i, 'label')), lead.get((i, 'label')), args.model])
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
