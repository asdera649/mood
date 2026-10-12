#!/usr/bin/env python3
"""
mood.py — одна нейросеть, два ответа: тональность и эмоции русского текста.
Тональность: негатив / нейтрально / позитив
Эмоции: радость, грусть, удивление, страх, злость (можно несколько сразу или ни одной)

Схема:
    текст -> BERT -> вектор текста -> голова 1 -> тональность (1 из 3 классов)
                                   -> голова 2 -> эмоции (5 независимых «да/нет»)

Использование:
    pip install -r requirements.txt
    python mood.py baseline # эталон без нейросети: число, которое надо побить
    python mood.py train # обучение -> папка mood_model/
    python mood.py predict "Ну спасибо, очень помогли" "Ура, всё получилось!"
    python mood.py predict --file messages.txt --csv result.csv # разбор файла, 1 сообщение = 1 строка
    python mood.py predict # интерактивный режим
"""
import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

# ----------------------------------------------------------------------------
# 1. Метки
# ----------------------------------------------------------------------------
SENT_LABELS = ["negative", "neutral", "positive"]
EMO_LABELS = ["joy", "sadness", "surprise", "fear", "anger"]  # порядок как в CEDR
SENT_RU = {"negative": "негатив", "neutral": "нейтрально", "positive": "позитив"}
EMO_RU = {"joy": "радость", "sadness": "грусть", "surprise": "удивление",
          "fear": "страх", "anger": "злость"}


def clean_text(text: str) -> str:
    """Одинаково чистим текст и при обучении, и при использовании."""
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # ссылки
    text = re.sub(r"@\w+", "@user", text)              # упоминания @ник -> @user
    return re.sub(r"\s+", " ", text).strip()           # переносы строк, двойные пробелы


# ----------------------------------------------------------------------------
# 2. Данные
# ----------------------------------------------------------------------------
def load_sentiment(seed=42):
    """RuSentiTweet: метки positive / neutral / negative / speech / skip.
    Классы speech (приветствия, поздравления) и skip (неясно, что это) выбрасываем."""
    from datasets import load_dataset
    ds = load_dataset("psytechlab/RuSentiTweet")  # сплиты: train / val / test
    out = {}
    for split in ("train", "val", "test"):
        texts, labels = [], []
        for row in ds[split]:
            if row["label"] not in SENT_LABELS:
                continue
            text = clean_text(row["text"])
            if text:
                texts.append(text)
                labels.append(SENT_LABELS.index(row["label"]))
        out[split] = (texts, labels)
    return out


def load_emotions(seed=42):
    """CEDR: 9410 предложений. Метка - список эмоций из EMO_LABELS; пустой список = "нет эмоции".
    Готовых val-сплита нет, поэтому 10% train отрезаем сами."""
    from datasets import load_dataset
    ds = load_dataset("sagteam/cedr_v1", "main")  # сплиты: train / test

    def convert(split):
        texts, labels = [], []
        for row in ds[split]:
            text = clean_text(row["text"])
            if not text:
                continue
            y = [0.0] * len(EMO_LABELS)
            for i in row["labels"]:
                y[i] = 1.0
            texts.append(text)
            labels.append(y)
        return texts, labels

    x, y = convert("train")
    tr_x, va_x, tr_y, va_y = train_test_split(x, y, test_size=0.1, random_state=seed)
    return {"train": (tr_x, tr_y), "val": (va_x, va_y), "test": convert("test")}


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        # padding здесь не делаем: дополним до самой длинной строки в батче (быстрее)
        self.enc = tokenizer(texts, truncation=True, max_length=max_len)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {"input_ids": self.enc["input_ids"][i], "attention_mask": self.enc["attention_mask"][i]}
        return item, self.labels[i]


def make_collate(tokenizer, label_dtype):
    def collate(batch):
        enc = tokenizer.pad([b[0] for b in batch], return_tensors="pt")
        y = torch.tensor([b[1] for b in batch], dtype=label_dtype)
        return enc["input_ids"], enc["attention_mask"], y
    return collate


# ----------------------------------------------------------------------------
# 3. Модель
# ----------------------------------------------------------------------------
class MoodModel(nn.Module):
    """Общий энкодер + две линейные головы."""

    def __init__(self, encoder, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.sent_head = nn.Linear(hidden, len(SENT_LABELS))
        self.emo_head = nn.Linear(hidden, len(EMO_LABELS))

    def forward(self, input_ids, attention_mask, task="both"):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)  # среднее по токенам (без паддинга)
        pooled = self.dropout(pooled)
        if task == "sent":
            return self.sent_head(pooled)
        if task == "emo":
            return self.emo_head(pooled)
        return self.sent_head(pooled), self.emo_head(pooled)


# ----------------------------------------------------------------------------
# 4. Вспомогательное для обучения
# ----------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def forever(loader):
    """Бесконечный итератор по DataLoader (меньший датасет просто крутится по кругу)."""
    while True:
        for batch in loader:
            yield batch


def class_weights(labels, n_classes):
    """Редким классам - больший вес, чтобы модель не отвечала всегда "нейтрально"."""
    counts = np.bincount(labels, minlength=n_classes).clip(min=1).astype(float)
    return torch.tensor(counts.sum() / (n_classes * counts), dtype=torch.float32)


@torch.no_grad()
def collect(model, loader, task, device):
    model.eval()
    logits, ys = [], []
    for ids, mask, y in loader:
        logits.append(model(ids.to(device), mask.to(device), task).float().cpu())
        ys.append(y)
    return torch.cat(logits), torch.cat(ys)


def tune_thresholds(probs, y):
    """Для каждой эмоции подбираем порог вероятности, при котором F1 на validation лучше всего."""
    probs, y = np.asarray(probs), np.asarray(y).astype(int)
    grid = np.arange(0.1, 0.91, 0.05)
    thresholds = []
    for j in range(y.shape[1]):
        scores = [f1_score(y[:, j], probs[:, j] >= t, zero_division=0) for t in grid]
        thresholds.append(float(grid[int(np.argmax(scores))]) if max(scores) > 0 else 0.5)
    return thresholds


def emo_predictions(probs, thresholds):
    return (np.asarray(probs) >= np.array(thresholds)).astype(int)


def describe(name, labels, names, multilabel=False):
    counts = np.array(labels).sum(0) if multilabel else np.bincount(labels, minlength=len(names))
    print(f"  {name:<18} {len(labels):>5} текстов | " + ", ".join(f"{n}={int(c)}" for n, c in zip(names, counts)))


# ----------------------------------------------------------------------------
# 5. Обучение
# ----------------------------------------------------------------------------
def train(args):
    set_seed(args.seed)
    device = get_device()
    print(f"Устройство: {device}")

    print("Загружаю датасеты...")
    sent, emo = load_sentiment(args.seed), load_emotions(args.seed)
    for split in ("train", "val", "test"):
        describe(f"тональность/{split}", sent[split][1], SENT_LABELS)
        describe(f"эмоции/{split}", emo[split][1], EMO_LABELS, multilabel=True)

    print(f"Загружаю модель {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = MoodModel(AutoModel.from_pretrained(args.model)).to(device)

    def loader(data, dtype, shuffle):
        ds = TextDataset(data[0], data[1], tokenizer, args.max_len)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          collate_fn=make_collate(tokenizer, dtype))

    s_train, s_val, s_test = [loader(sent[k], torch.long, k == "train") for k in ("train", "val", "test")]
    e_train, e_val, e_test = [loader(emo[k], torch.float32, k == "train") for k in ("train", "val", "test")]

    sent_w = class_weights(sent["train"][1], len(SENT_LABELS)).to(device)
    heads = list(model.sent_head.parameters()) + list(model.emo_head.parameters())
    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.lr},  # BERT дообучаем аккуратно
        {"params": heads, "lr": args.head_lr},                  # новые головы учим быстрее
    ], weight_decay=0.01)

    steps_per_epoch = max(len(s_train), len(e_train))
    total, warmup = steps_per_epoch * args.epochs, int(0.1 * steps_per_epoch * args.epochs)

    def lr_scale(step):  # разгон 10% шагов, потом линейно до нуля
        return step / max(1, warmup) if step < warmup else max(0.0, (total - step) / max(1, total - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        it_s, it_e = forever(s_train), forever(e_train)
        running = 0.0
        for _ in tqdm(range(steps_per_epoch), desc=f"эпоха {epoch}/{args.epochs}"):
            ids, mask, y = next(it_s)  # батч с тональностью
            loss_s = F.cross_entropy(model(ids.to(device), mask.to(device), "sent"), y.to(device), weight=sent_w)
            ids, mask, y = next(it_e)  # батч с эмоциями
            loss_e = F.binary_cross_entropy_with_logits(model(ids.to(device), mask.to(device), "emo"), y.to(device))
            loss = loss_s + loss_e
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()

        logits, y = collect(model, s_val, "sent", device)
        f1_s = f1_score(y.numpy(), logits.argmax(1).numpy(), average="macro")
        logits, y = collect(model, e_val, "emo", device)
        probs = torch.sigmoid(logits).numpy()
        thresholds = tune_thresholds(probs, y.numpy())
        f1_e = f1_score(y.numpy().astype(int), emo_predictions(probs, thresholds), average="macro", zero_division=0)
        print(f"  loss {running / steps_per_epoch:.3f} | val macro-F1: тональность {f1_s:.3f}, эмоции {f1_e:.3f}")

        if (f1_s + f1_e) / 2 > best:  # сохраняем лучшую эпоху
            best = (f1_s + f1_e) / 2
            torch.save(model.state_dict(), out_dir / "model.pt")
            tokenizer.save_pretrained(out_dir)
            model.encoder.config.save_pretrained(out_dir)
            meta = {"base_model": args.model, "max_len": args.max_len, "thresholds": thresholds, "epoch": epoch}
            (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            print("  ✔ лучшая эпоха, модель сохранена")

    # Финальная проверка на тесте, который модель ни разу не видела
    model.load_state_dict(torch.load(out_dir / "model.pt", map_location=device))
    thresholds = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))["thresholds"]
    print("\n=== ТЕСТ: тональность (RuSentiTweet) ===")
    logits, y = collect(model, s_test, "sent", device)
    print(classification_report(y.numpy(), logits.argmax(1).numpy(), digits=3,
                                target_names=[SENT_RU[l] for l in SENT_LABELS], zero_division=0))
    print("=== ТЕСТ: эмоции (CEDR) ===")
    logits, y = collect(model, e_test, "emo", device)
    print(classification_report(y.numpy().astype(int), emo_predictions(torch.sigmoid(logits).numpy(), thresholds), digits=3,
                                target_names=[EMO_RU[l] for l in EMO_LABELS], zero_division=0))
    print(f"Готово. Модель в папке {out_dir}/. Пробуй: python mood.py predict \"Ура, всё получилось!\"")


# ----------------------------------------------------------------------------
# 5б. Эталон без нейросети
# ----------------------------------------------------------------------------
def baseline(args):
    """TF-IDF + логистическая регрессия. Запустить до train: это число, которое BERT обязан побить."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import make_pipeline

    sent, emo = load_sentiment(args.seed), load_emotions(args.seed)

    def tfidf():  # буквенные n-граммы переживают опечатки, сленг и мат лучше целых слов
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)

    clf = make_pipeline(tfidf(), LogisticRegression(C=3, max_iter=1000, class_weight="balanced"))
    clf.fit(sent["train"][0], sent["train"][1])
    print("=== ЭТАЛОН, тональность (тест) ===")
    print(classification_report(sent["test"][1], clf.predict(sent["test"][0]), digits=3,
                                target_names=[SENT_RU[l] for l in SENT_LABELS], zero_division=0))

    clf = make_pipeline(tfidf(), OneVsRestClassifier(LogisticRegression(C=3, max_iter=1000)))
    clf.fit(emo["train"][0], np.array(emo["train"][1]).astype(int))
    thresholds = tune_thresholds(clf.predict_proba(emo["val"][0]), emo["val"][1])
    print("=== ЭТАЛОН, эмоции (тест) ===")
    pred = emo_predictions(clf.predict_proba(emo["test"][0]), thresholds)
    print(classification_report(np.array(emo["test"][1]).astype(int), pred, digits=3,
                                target_names=[EMO_RU[l] for l in EMO_LABELS], zero_division=0))


# ----------------------------------------------------------------------------
# 6. Использование обученной модели
# ----------------------------------------------------------------------------
def load_model(model_dir, device):
    model_dir = Path(model_dir)
    if not (model_dir / "model.pt").exists():
        raise SystemExit(f"Не нашёл {model_dir}/model.pt — сначала запусти: python mood.py train")
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_dir))  # веса подгрузим из model.pt
    model = MoodModel(encoder)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    return model.to(device).eval(), tokenizer, meta


@torch.no_grad()
def analyze(texts, model, tokenizer, meta, device, batch_size=32):
    results = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tokenizer([clean_text(t) for t in chunk], truncation=True, max_length=meta["max_len"],
                        padding=True, return_tensors="pt")
        s_logits, e_logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device), "both")
        s_probs = F.softmax(s_logits, dim=-1).cpu().numpy()
        e_probs = torch.sigmoid(e_logits).cpu().numpy()
        for text, sp, ep in zip(chunk, s_probs, e_probs):
            k = int(sp.argmax())
            results.append({
                "text": text,
                "sentiment": SENT_LABELS[k],
                "sent_conf": float(sp[k]),
                "sent_probs": {l: float(p) for l, p in zip(SENT_LABELS, sp)},
                "emo_probs": {l: float(p) for l, p in zip(EMO_LABELS, ep)},
                # эмоции, чья вероятность выше подобранного порога
                "emotions": {l: float(p) for l, p, t in zip(EMO_LABELS, ep, meta["thresholds"]) if p >= t},
            })
    return results


def show(r, min_conf):
    sent = SENT_RU[r["sentiment"]]
    if r["sent_conf"] < min_conf:
        sent = f"не уверен (похоже на «{sent}»)"
    emo = ", ".join(f"{EMO_RU[k]} {p:.2f}" for k, p in r["emotions"].items()) or "нет выраженной эмоции"
    print(f"• {r['text']}\n    тональность: {sent} ({r['sent_conf']:.2f})\n    эмоции: {emo}")


def summarize(results, min_conf):
    n = len(results)
    cnt = Counter(r["sentiment"] for r in results)
    unsure = sum(r["sent_conf"] < min_conf for r in results)
    print(f"\nСообщений: {n}  (не уверен в тональности: {unsure})")
    for l in SENT_LABELS:
        print(f"  {SENT_RU[l]:<11} {cnt[l]:>5}  {100 * cnt[l] / n:>3.0f}%")
    emo_cnt = Counter(k for r in results for k in r["emotions"])
    print("Эмоции: " + (", ".join(f"{EMO_RU[k]} {emo_cnt[k]}" for k in EMO_LABELS if emo_cnt[k]) or "не найдено"))
    print("\nСамые негативные:")
    for r in sorted(results, key=lambda r: r["sent_probs"]["negative"], reverse=True)[:5]:
        print(f"  {r['sent_probs']['negative']:.2f}  {r['text'][:100]}")


def save_csv(results, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig - чтобы Excel не ломал кириллицу
        w = csv.writer(f)
        w.writerow(["text", "sentiment", "confidence", *[f"p_{l}" for l in SENT_LABELS],
                    "emotions", *[f"p_{l}" for l in EMO_LABELS]])
        for r in results:
            w.writerow([r["text"], SENT_RU[r["sentiment"]], f"{r['sent_conf']:.3f}",
                        *[f"{r['sent_probs'][l]:.3f}" for l in SENT_LABELS],
                        ", ".join(EMO_RU[k] for k in r["emotions"]),
                        *[f"{r['emo_probs'][l]:.3f}" for l in EMO_LABELS]])
    print(f"Таблица сохранена: {path}")


def predict(args):
    device = get_device()
    model, tokenizer, meta = load_model(args.model_dir, device)

    def run(texts):
        return analyze(texts, model, tokenizer, meta, device)

    if args.file:
        lines = Path(args.file).read_text(encoding="utf-8-sig").splitlines()
        results = run([ln.strip() for ln in lines if ln.strip()])
        summarize(results, args.min_conf)
        if args.csv:
            save_csv(results, args.csv)
    elif args.texts:
        for r in run(args.texts):
            show(r, args.min_conf)
    else:
        print("Пиши сообщение и жми Enter (пустая строка — выход).")
        while True:
            try:
                text = input("> ").strip()
            except EOFError:
                break
            if not text:
                break
            show(run([text])[0], args.min_conf)


# ----------------------------------------------------------------------------
# 7. Командная строка
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Тональность и эмоции русского текста одной нейросетью")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("baseline", help="эталон без нейросети (TF-IDF + логрегрессия); запусти до train")
    b.add_argument("--seed", type=int, default=42)
    b.set_defaults(func=baseline)

    t = sub.add_parser("train", help="обучить модель")
    t.add_argument("--model", default="cointegrated/rubert-tiny2",
                   help="какой BERT дообучать; для качества лучше DeepPavlov/rubert-base-cased (нужна GPU)")
    t.add_argument("--epochs", type=int, default=5)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-4,
                   help="скорость обучения BERT: 1e-4 для rubert-tiny2, для base-модели ставь 3e-5")
    t.add_argument("--head-lr", type=float, default=1e-3, help="скорость обучения новых голов")
    t.add_argument("--max-len", type=int, default=64, help="максимум токенов на сообщение")
    t.add_argument("--out", default="mood_model")
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=train)

    q = sub.add_parser("predict", help="применить обученную модель")
    q.add_argument("texts", nargs="*", help="тексты в кавычках; без них — интерактивный режим")
    q.add_argument("--model-dir", default="mood_model")
    q.add_argument("--file", help="txt-файл: одно сообщение на строку")
    q.add_argument("--csv", help="куда сохранить таблицу результатов (вместе с --file)")
    q.add_argument("--min-conf", type=float, default=0.5,
                   help="если уверенность в тональности ниже — пишем «не уверен»")
    q.set_defaults(func=predict)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
