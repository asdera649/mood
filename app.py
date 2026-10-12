#!/usr/bin/env python3
"""
app.py — локальный интерфейс к модели из mood.py: тональность + эмоции, графики.

Запуск:
    pip install -r requirements_app.txt
    python run_app.py
"""
import html
import io
import random
import re
import warnings
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import mood  # mood.py должен лежать рядом с app.py

st.set_page_config(page_title="Настроение текста", page_icon="", layout="wide")

# В новых Streamlit use_container_width заменили на width="stretch"
_VER = tuple(int(x) for x in st.__version__.split(".")[:2])
STRETCH = {"width": "stretch"} if _VER >= (1, 65) else {"use_container_width": True}

# ----------------------------------------------------------------------------
# Оформление
# ----------------------------------------------------------------------------
SENT_COLORS = {"negative": "#F43F5E", "neutral": "#94A3B8", "positive": "#10B981"}
SENT_EMOJI = {"negative": "😞", "neutral": "😐", "positive": "🙂"}
EMO_COLORS = {"joy": "#F59E0B", "sadness": "#3B82F6", "surprise": "#A855F7", "fear": "#14B8A6", "anger": "#EF4444"}
EMO_EMOJI = {"joy": "😊", "sadness": "😢", "surprise": "😮", "fear": "😨", "anger": "😠"}
ACCENT = "#8B5CF6"

CSS = """
<style>
.block-container {padding-top: 3rem; max-width: 1280px;}
.hero {background: linear-gradient(120deg,#6366F1 0%,#8B5CF6 50%,#EC4899 100%); border-radius: 18px;
       padding: 22px 28px; margin-bottom: 18px;}
.hero h1 {margin: 0; padding: 0; font-size: 2rem; color: #fff !important;}
.hero p {margin: 6px 0 0; color: #fff !important; opacity: .92; font-size: 1rem;}
.card {border: 1px solid rgba(128,128,128,.28); background: rgba(128,128,128,.08); border-radius: 14px;
       padding: 14px 18px; min-height: 122px;}
.card .t {font-size: .74rem; letter-spacing: .07em; text-transform: uppercase; opacity: .65;}
.card .v {font-size: 1.75rem; font-weight: 700; line-height: 1.25; margin: 2px 0;}
.card .s {font-size: .85rem; opacity: .72;}
.chip {display: inline-block; padding: 2px 8px; margin: 2px 2px; border-radius: 8px; font-size: 1.08rem;}
[data-testid="stAppDeployButton"], .stDeployButton {display: none;}
.legend span {display: inline-block; padding: 1px 9px; border-radius: 6px; margin-right: 6px; font-size: .82rem;}
</style>
"""

EXAMPLES = [
    ("🙂 Радость", "Ура! Наконец-то всё заработало, спасибо огромное!"),
    ("😠 Злость", "Опять списали деньги, и никто не отвечает. Это просто бесит!"),
    ("😨 Страх", "Мне страшно, что я не успею сдать работу вовремя"),
    ("😢 Грусть", "Очень грустно, что так всё закончилось"),
    ("😮 Удивление", "Ничего себе, я совсем не ожидал такого результата!"),
    ("🙃 Сарказм", "Ну спасибо, очень помогли, прямо выручили"),
]


def card(title, value, sub="", color=None):
    style = f' style="color:{color}"' if color else ""
    return (f'<div class="card"><div class="t">{title}</div><div class="v"{style}>{value}</div>'
            f'<div class="s">{sub}</div></div>')


def score_color(score):
    return SENT_COLORS["positive"] if score > 0.2 else SENT_COLORS["negative"] if score < -0.2 else SENT_COLORS["neutral"]


# ----------------------------------------------------------------------------
# Модель
# ----------------------------------------------------------------------------
def resolve_model_dir(raw):
    p = Path((raw or "mood_model").strip()).expanduser()
    for cand in (p, p / "mood_model"):  # на случай, если архив распаковался во вложенную папку
        if (cand / "model.pt").exists() and (cand / "meta.json").exists():
            return str(cand.resolve())
    return None


@st.cache_resource(show_spinner="Загружаю модель…")
def load_bundle(model_dir):
    device = mood.get_device()
    model, tokenizer, meta = mood.load_model(model_dir, device)
    return {"model": model, "tokenizer": tokenizer, "meta": meta, "device": str(device),
            "n_params": sum(p.numel() for p in model.parameters())}


def run_model(texts, model_dir, progress=None, chunk=64):
    b = load_bundle(model_dir)
    device = mood.get_device()
    out = []
    for i in range(0, len(texts), chunk):
        out += mood.analyze(texts[i:i + chunk], b["model"], b["tokenizer"], b["meta"], device, batch_size=chunk)
        if progress is not None:
            done = min(i + chunk, len(texts))
            progress.progress(done / len(texts), text=f"Обработано {done} из {len(texts)}")
    return out


@st.cache_data(show_spinner=False)
def analyze_one(text, model_dir):
    return run_model([text], model_dir)[0]


@st.cache_data(show_spinner=False)
def explain_words(text, model_dir, max_words=60):
    """Метод "вычеркни слово": на сколько меняется индекс настроения, если слово убрать.
    Индекс = P(позитив) − P(негатив). Вклад > 0: слово тянет текст к позитиву, < 0: к негативу."""
    words = text.split()
    if len(words) < 2 or len(words) > max_words:
        return None
    index = lambda r: r["sent_probs"]["positive"] - r["sent_probs"]["negative"]
    base = index(run_model([text], model_dir)[0])
    variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    return [(w, base - index(r)) for w, r in zip(words, run_model(variants, model_dir))]


def to_frame(results):
    rows = []
    for r in results:
        row = {"text": r["text"], "sentiment": r["sentiment"], "confidence": r["sent_conf"]}
        row.update({f"p_{l}": r["sent_probs"][l] for l in mood.SENT_LABELS})
        row.update({f"e_{l}": r["emo_probs"][l] for l in mood.EMO_LABELS})
        row.update({f"has_{l}": l in r["emotions"] for l in mood.EMO_LABELS})
        row["score"] = row["p_positive"] - row["p_negative"]
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Графики
# ----------------------------------------------------------------------------
def style(fig, title=None, height=300, margin=None, **layout):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15), x=0, xanchor="left") if title else None,
        height=height, margin=margin or dict(l=10, r=10, t=48 if title else 10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", **layout)
    return fig


def show(fig, key):
    st.plotly_chart(fig, key=key, config={"displayModeBar": False}, **STRETCH)


def fig_sentiment_bar(probs, pred):
    labels = mood.SENT_LABELS
    fig = go.Figure(go.Bar(
        x=[probs[l] for l in labels], y=[f"{mood.SENT_RU[l]}" for l in labels], orientation="h",
        marker=dict(color=[SENT_COLORS[l] for l in labels], opacity=[1.0 if l == pred else 0.4 for l in labels]),
        text=[f"{probs[l]:.0%}" for l in labels], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_xaxes(range=[0, 1.15], visible=False)
    fig.update_yaxes(autorange="reversed")
    return style(fig, "Тональность: вероятности", 280, showlegend=False)


def fig_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number=dict(valueformat="+.2f", font=dict(size=40)),
        gauge=dict(axis=dict(range=[-1, 1], tickvals=[-1, -0.5, 0, 0.5, 1]),
                   bar=dict(color=score_color(score), thickness=0.35), borderwidth=0,
                   steps=[dict(range=[-1, -0.33], color="rgba(244,63,94,.18)"),
                          dict(range=[-0.33, 0.33], color="rgba(148,163,184,.18)"),
                          dict(range=[0.33, 1], color="rgba(16,185,129,.18)")])))
    return style(fig, "Индекс настроения", 280, margin=dict(l=30, r=30, t=70, b=20))


def fig_emotion_radar(probs, thresholds):
    labels = mood.EMO_LABELS
    names = [f"{EMO_EMOJI[l]} {mood.EMO_RU[l]}" for l in labels]
    p = [probs[l] for l in labels]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=p + p[:1], theta=names + names[:1], mode="lines", fill="toself", name="вероятность",
                                  line=dict(color=ACCENT, width=2), fillcolor="rgba(139,92,246,.30)",
                                  hovertemplate="%{theta}: %{r:.0%}<extra></extra>"))
    fig.add_trace(go.Scatterpolar(r=list(thresholds) + list(thresholds[:1]), theta=names + names[:1], mode="lines", name="порог",
                                  line=dict(color="#F43F5E", width=1.6, dash="dash"),
                                  hovertemplate="порог: %{r:.0%}<extra></extra>"))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                                 radialaxis=dict(range=[0, 1], tickvals=[0.5, 1], tickformat=".0%", tickfont=dict(size=9))),
                      legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"))
    return style(fig, "Эмоции", 280, margin=dict(l=40, r=40, t=48, b=30))


def fig_donut(df):
    counts = df["sentiment"].value_counts()
    fig = go.Figure(go.Pie(
        labels=[mood.SENT_RU[l] for l in mood.SENT_LABELS], values=[int(counts.get(l, 0)) for l in mood.SENT_LABELS],
        hole=0.62, sort=False, marker=dict(colors=[SENT_COLORS[l] for l in mood.SENT_LABELS]),
        textinfo="percent", hovertemplate="%{label}: %{value} (%{percent})<extra></extra>"))
    fig.add_annotation(text=f"<b>{len(df)}</b><br>сообщений", showarrow=False, font=dict(size=15))
    return style(fig, "Тональность", 300, legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"))


def fig_emotion_counts(df):
    vals = [int(df[f"has_{l}"].sum()) for l in mood.EMO_LABELS]
    fig = go.Figure(go.Bar(
        x=[f"{EMO_EMOJI[l]}<br>{mood.EMO_RU[l]}" for l in mood.EMO_LABELS], y=vals,
        marker_color=[EMO_COLORS[l] for l in mood.EMO_LABELS],
        text=[f"{v} ({v / len(df):.0%})" for v in vals], textposition="outside", cliponaxis=False,
        hovertemplate="%{x}: %{y}<extra></extra>"))
    fig.update_xaxes(tickangle=0)
    fig.update_yaxes(visible=False, range=[0, max(vals + [1]) * 1.2])
    return style(fig, "Эмоции в сообщениях", 300)


def fig_heatmap(df):
    z = [[int(((df["sentiment"] == s) & df[f"has_{e}"]).sum()) for e in mood.EMO_LABELS] for s in mood.SENT_LABELS]
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{EMO_EMOJI[e]}<br>{mood.EMO_RU[e]}" for e in mood.EMO_LABELS],
        y=[f"{SENT_EMOJI[s]} {mood.SENT_RU[s]}" for s in mood.SENT_LABELS],
        colorscale=[[0, "rgba(139,92,246,0.06)"], [1, "rgba(139,92,246,0.95)"]], showscale=False,
        texttemplate="%{z}", xgap=3, ygap=3, hovertemplate="%{y} + %{x}: %{z}<extra></extra>"))
    fig.update_xaxes(tickangle=0)
    fig.update_yaxes(autorange="reversed")
    return style(fig, "Эмоции и тональность вместе", 300)


UNIT_DAT = {"час": "часам", "день": "дням", "неделя": "неделям", "месяц": "месяцам"}


def fig_timeline(per, flagged, unit):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=per.index, y=per["n"], name="сообщений", marker_color="rgba(148,163,184,.35)", secondary_y=True,
                hovertemplate="%{y} сообщений<extra></extra>")
    fig.add_scatter(x=per.index, y=per["score"], name="индекс настроения", mode="lines+markers",
                    line=dict(color=ACCENT, width=3), marker=dict(size=7), secondary_y=False,
                    hovertemplate="индекс %{y:+.2f}<extra></extra>")
    bad = per[flagged]
    if len(bad):
        fig.add_scatter(x=bad.index, y=bad["score"], mode="markers", name="всплеск негатива", secondary_y=False,
                        marker=dict(color="#F43F5E", size=17, symbol="circle-open", line=dict(width=3)),
                        hovertemplate="всплеск негатива<extra></extra>")
    fig.update_yaxes(range=[-1, 1], title_text="индекс", zeroline=True, secondary_y=False)
    fig.update_yaxes(title_text="сообщений", showgrid=False, tickmode="auto", nticks=5, secondary_y=True)
    return style(fig, f"Индекс настроения по {UNIT_DAT[unit]}", 340, legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"))


def fig_shares(per, unit):
    fig = go.Figure()
    for l in mood.SENT_LABELS:
        fig.add_bar(x=per.index, y=per[f"share_{l}"], name=mood.SENT_RU[l], marker_color=SENT_COLORS[l],
                    hovertemplate=f"{mood.SENT_RU[l]}: " + "%{y:.0%}<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return style(fig, f"Доли тональностей по {UNIT_DAT[unit]}", 340, legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"))


def fig_words(words, color, title):
    if not words:
        return None
    words = words[::-1]
    fig = go.Figure(go.Bar(x=[c for _, c in words], y=[w for w, _ in words], orientation="h", marker_color=color,
                           text=[c for _, c in words], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: в %{x} сообщ.<extra></extra>"))
    fig.update_xaxes(visible=False)
    return style(fig, title, 340)


# ----------------------------------------------------------------------------
# Данные
# ----------------------------------------------------------------------------
STOP = set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от
меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там
потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под
будет ж тогда кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем
всех никогда можно при наконец два об другой хоть после над больше тот через эти нас про всего них какая много разве
три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между это
очень просто какие которые который которая также свои своё всё""".split())


def top_words(texts, n=12):
    """Сколько сообщений содержит слово (без лемматизации, поэтому "ошибка" и "ошибки" считаются отдельно)."""
    cnt = Counter()
    for t in texts:
        cnt.update({w for w in re.findall(r"[а-яёa-z]{3,}", t.lower()) if w not in STOP})
    return cnt.most_common(n)


def to_datetime(s):
    """Даты вида 2026-10-01 читаем как ISO (год-месяц-день), остальные (01.10.2026) как день.месяц.год."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            iso = pd.to_datetime(s, errors="coerce", format="ISO8601")
            if iso.notna().mean() >= 0.8:
                return iso
        except (ValueError, TypeError):
            pass
        return pd.to_datetime(s, errors="coerce", dayfirst=True)


def bucket(dt, unit):
    if unit == "час":
        return dt.dt.floor("60min")
    if unit == "день":
        return dt.dt.normalize()
    if unit == "неделя":
        return dt.dt.to_period("W").dt.start_time
    return dt.dt.to_period("M").dt.start_time


def auto_unit(dt):
    span = (dt.max() - dt.min()).days
    return "час" if span < 2 else "день" if span <= 120 else "неделя"


def aggregate(df, unit):
    d = df.dropna(subset=["dt"]).copy()
    d["period"] = bucket(d["dt"], unit)
    shares = pd.crosstab(d["period"], d["sentiment"], normalize="index").reindex(columns=mood.SENT_LABELS, fill_value=0)
    per = pd.DataFrame({"n": d.groupby("period").size(), "score": d.groupby("period")["score"].mean()})
    for l in mood.SENT_LABELS:
        per[f"share_{l}"] = shares[l]
    return per


def find_spikes(per, min_n=5):
    """Всплеск = доля негатива заметно выше обычной (на 10+ п.п. и больше чем 1.5 стандартных отклонения)."""
    ok = per["n"] >= min_n
    neg = per["share_negative"]
    if ok.sum() < 4:
        return pd.Series(False, index=per.index), None
    mu, sd = neg[ok].mean(), neg[ok].std(ddof=0)
    return ok & (neg > mu + max(0.10, 1.5 * sd)), mu


TEXT_HINTS = ("text", "message", "msg", "comment", "review", "сообщ", "текст", "коммент", "отзыв")
DATE_HINTS = ("date", "time", "created", "timestamp", "дата", "время")


def guess_text_col(df):
    for c in df.columns:
        if any(h in str(c).lower() for h in TEXT_HINTS):
            return c
    cand = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    return max(cand, key=lambda c: df[c].astype(str).str.len().mean()) if cand else df.columns[0]


def guess_date_col(df, exclude):
    for c in df.columns:
        if c != exclude and any(h in str(c).lower() for h in DATE_HINTS):
            return c
    for c in df.columns:
        if c != exclude and not pd.api.types.is_numeric_dtype(df[c]) and to_datetime(df[c]).notna().mean() > 0.8:
            return c
    return None


def read_upload(up):
    raw = up.getvalue()
    for enc in ("utf-8-sig", "cp1251"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("не удалось определить кодировку (нужна UTF-8 или Windows-1251)")
    if up.name.lower().endswith(".csv"):
        df = pd.read_csv(io.StringIO(text))
        if df.shape[1] == 1:  # Excel по-русски часто сохраняет CSV с разделителем «;»
            first = text.splitlines()[0]
            for sep in (";", "\t", "|"):
                if sep in first:
                    return pd.read_csv(io.StringIO(text), sep=sep)
        return df
    return pd.DataFrame({"text": [ln.strip() for ln in text.splitlines() if ln.strip()]})


DEMO = {
    "positive": ["Спасибо, всё отлично работает!", "Обновление супер, очень доволен", "Ура, наконец-то заработало, спасибо огромное!",
                 "Классная идея, давно этого ждал", "Поддержка ответила за минуту, вы молодцы",
                 "Очень удобный интерфейс, всё интуитивно понятно", "Круто, что вы прислушались к пользователям",
                 "Приложение стало работать заметно быстрее, супер!", "Спасибо за быстрое решение проблемы",
                 "Обожаю эту функцию, пользуюсь каждый день", "Отличное обновление, так держать!", "Всё получилось, вы лучшие!"],
    "neutral": ["Подскажите, где найти настройки профиля?", "Когда выйдет следующее обновление?",
                "Сегодня обычный день, пока без изменений", "Как сменить язык интерфейса?", "Сервер доступен, проверил только что",
                "Версия приложения 2.4, Android", "Нужна инструкция по восстановлению пароля", "Где посмотреть историю заказов?",
                "Отправил вам письмо на почту поддержки", "Работает с утра, как обычно", "Какая сейчас версия сайта?",
                "Принято, жду ответа"],
    "negative": ["Опять всё сломалось, меня это просто бесит!", "Деньги списали, а услугу не дали. Ужасный сервис",
                 "Невозможно пользоваться, постоянные вылеты", "Мне очень грустно, что вы убрали старую функцию",
                 "Боюсь, что мой аккаунт взломали, помогите", "Поддержка молчит уже третий день, это возмутительно",
                 "Разочарован, ожидал гораздо большего", "Ненавижу эту рекламу, невозможно терпеть",
                 "Приложение тормозит, пользоваться невозможно", "Мне страшно, что пропадут все мои данные"],
    "spike": ["После обновления вообще ничего не работает, бесит!", "Обновление всё сломало, верните старую версию",
              "Опять вылетает после обновления, сколько можно", "Потерял все данные после обновления, это ужас",
              "Зачем вы выпустили это обновление? Одни баги", "После апдейта не могу войти, поддержка не отвечает",
              "Разработчики, вы вообще тестируете свои обновления?", "Мне страшно открывать приложение после этого обновления"],
}


def make_demo(days=14, seed=7):
    """Две недели "чата поддержки"; на 9-й день после "неудачного обновления" негатив резко растёт."""
    rnd = random.Random(seed)
    start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days - 1)
    spike_day, rows = 8, []
    for d in range(days):
        spike = d == spike_day
        weights = [0.55, 0.12, 0.33] if spike else [0.17, 0.38, 0.45]  # negative / positive / neutral
        for _ in range(rnd.randint(25, 40) + (30 if spike else 0)):
            kind = rnd.choices(["negative", "positive", "neutral"], weights=weights)[0]
            pool = DEMO["spike"] + DEMO["negative"] if (spike and kind == "negative") else DEMO[kind]
            ts = start + timedelta(days=d, hours=rnd.randint(8, 23), minutes=rnd.randint(0, 59))
            rows.append((ts, rnd.choice(pool)))
    return pd.DataFrame(rows, columns=["date", "text"])


# ----------------------------------------------------------------------------
# Вкладка 1: один текст
# ----------------------------------------------------------------------------
def word_chips(pairs):
    scale = max(0.15, max(abs(d) for _, d in pairs))
    spans = []
    for w, d in pairs:
        if abs(d) < 0.01:
            bg = "transparent"
        else:
            rgb = "16,185,129" if d > 0 else "244,63,94"
            bg = f"rgba({rgb},{min(1.0, abs(d) / scale) * 0.8:.2f})"
        spans.append(f'<span class="chip" style="background:{bg}" title="вклад {d:+.2f}">{html.escape(w)}</span>')
    return "".join(spans)


def tab_single(model_dir, bundle, min_conf):
    st.markdown("##### Напиши сообщение или выбери пример")
    for col, (label, text) in zip(st.columns(len(EXAMPLES)), EXAMPLES):
        col.button(label, key=f"ex_{label}", on_click=lambda t=text: st.session_state.update(single_text=t), **STRETCH)
    st.text_area("Сообщение", key="single_text", height=110, label_visibility="collapsed",
                 placeholder="Например: Ну спасибо, очень помогли…")
    text = st.session_state.get("single_text", "").strip()
    if not text:
        st.info("Нажми на пример или напиши своё сообщение. Результат появится сразу (Ctrl+Enter).")
        return

    r = analyze_one(text, model_dir)
    probs, pred, conf = r["sent_probs"], r["sentiment"], r["sent_conf"]
    score = probs["positive"] - probs["negative"]
    emos = sorted(r["emotions"].items(), key=lambda kv: -kv[1])

    k1, k2, k3 = st.columns(3)
    with k1:
        if conf < min_conf:
            st.markdown(card("Тональность", "не уверен", f"похоже на «{mood.SENT_RU[pred]}», уверенность {conf:.0%}",
                             SENT_COLORS["neutral"]), unsafe_allow_html=True)
        else:
            st.markdown(card("Тональность", f"{mood.SENT_RU[pred]}", f"уверенность {conf:.0%}",
                             SENT_COLORS[pred]), unsafe_allow_html=True)
    with k2:
        if emos:
            top, p = emos[0]
            rest = ", ".join(f"{mood.EMO_RU[k]} {v:.0%}" for k, v in emos[1:]) or "других нет"
            st.markdown(card("Главная эмоция", f"{mood.EMO_RU[top]}", f"{p:.0%} · ещё: {rest}",
                             EMO_COLORS[top]), unsafe_allow_html=True)
        else:
            st.markdown(card("Эмоции", "не выражены", "ни одна не прошла свой порог"), unsafe_allow_html=True)
    with k3:
        st.markdown(card("Индекс настроения", f"{score:+.2f}", "от −1 (негатив) до +1 (позитив)", score_color(score)),
                    unsafe_allow_html=True)

    st.write("")
    g1, g3 = st.columns([1, 1])
    with g1, st.container(border=True):
        show(fig_sentiment_bar(probs, pred), "sent_bar")
    # with g2, st.container(border=True):
    #     show(fig_gauge(score), "gauge")
    with g3, st.container(border=True):
        show(fig_emotion_radar(r["emo_probs"], bundle["meta"]["thresholds"]), "radar")
    st.caption("Радар эмоций: если фиолетовая область выходит за красный пунктир, эмоция засчитана.")

    n_tok = len(bundle["tokenizer"](mood.clean_text(text))["input_ids"])
    if n_tok > bundle["meta"]["max_len"]:
        st.warning(f"В тексте {n_tok} токенов, модель видит только первые {bundle['meta']['max_len']}. Остальное она не читает.")

    with st.container(border=True):
        st.markdown("##### Какие слова повлияли на вердикт")
        pairs = explain_words(text, model_dir)
        if pairs is None:
            st.caption("Объяснение строится для текстов от 2 до 60 слов.")
        else:
            st.markdown('<div class="legend"><span style="background:rgba(16,185,129,.55)">тянет к позитиву</span>'
                        '<span style="background:rgba(244,63,94,.55)">тянет к негативу</span>'
                        '<span style="opacity:.6">без цвета: почти не влияет</span></div>', unsafe_allow_html=True)
            st.markdown(f"<div style='margin-top:10px'>{word_chips(pairs)}</div>", unsafe_allow_html=True)
            st.caption("Метод «вычеркни слово»: пересчитываем индекс настроения без каждого слова по очереди. "
                       "Наведи курсор на слово, чтобы увидеть вклад.")

    with st.expander("Все числа"):
        rows = [{"": f"{mood.SENT_RU[l]}", "вероятность": probs[l], "порог": None} for l in mood.SENT_LABELS]
        rows += [{"": f"{mood.EMO_RU[l]}", "вероятность": r["emo_probs"][l], "порог": t}
                 for l, t in zip(mood.EMO_LABELS, bundle["meta"]["thresholds"])]
        st.dataframe(pd.DataFrame(rows), hide_index=True, **STRETCH,
                     column_config={"вероятность": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                                    "порог": st.column_config.NumberColumn(format="%.2f")})


# ----------------------------------------------------------------------------
# Вкладка 2: пульс чата
# ----------------------------------------------------------------------------
def pick_columns(raw):
    raw = raw.copy()
    raw.columns = [str(c) for c in raw.columns]
    cols = list(raw.columns)
    if len(cols) == 1:
        return raw.rename(columns={cols[0]: "text"})
    t_guess = guess_text_col(raw)
    d_guess = guess_date_col(raw, t_guess)
    c1, c2 = st.columns(2)
    text_col = c1.selectbox("Колонка с текстом", cols, index=cols.index(t_guess))
    opts = ["— нет —"] + cols
    date_col = c2.selectbox("Колонка с датой (для графиков по времени)", opts, index=opts.index(d_guess) if d_guess else 0)
    out = pd.DataFrame({"text": raw[text_col]})
    if date_col != "— нет —":
        out["date"] = raw[date_col]
    return out


def table_view(df, min_conf):
    sent = [f"{SENT_EMOJI[s]} {mood.SENT_RU[s]}" if c >= min_conf else "не уверен" for s, c in zip(df["sentiment"], df["confidence"])]
    emo = [", ".join(f"{EMO_EMOJI[l]} {mood.EMO_RU[l]}" for l in mood.EMO_LABELS if row[f"has_{l}"]) for _, row in df.iterrows()]
    v = pd.DataFrame({"Сообщение": df["text"].values, "Тональность": sent, "Уверенность": df["confidence"].values,
                      "Индекс": df["score"].values, "Эмоции": emo}, index=df.index)
    if "dt" in df:
        v.insert(0, "Дата", df["dt"])
    return v


def export_csv(df):
    out = pd.DataFrame({"text": df["text"], "sentiment": df["sentiment"].map(mood.SENT_RU), "confidence": df["confidence"].round(3)})
    if "dt" in df:
        out.insert(0, "date", df["dt"])
    for l in mood.SENT_LABELS:
        out[f"p_{l}"] = df[f"p_{l}"].round(3)
    out["score"] = df["score"].round(3)
    out["emotions"] = [", ".join(mood.EMO_RU[l] for l in mood.EMO_LABELS if row[f"has_{l}"]) for _, row in df.iterrows()]
    for l in mood.EMO_LABELS:
        out[f"p_{l}"] = df[f"e_{l}"].round(3)
    return out.to_csv(index=False).encode("utf-8-sig")  # utf-8-sig, чтобы Excel не ломал кириллицу


def tab_batch(model_dir, min_conf):
    st.markdown("##### Загрузи пачку сообщений и посмотри картину целиком")
    src = st.radio("Источник", ["Файл (.txt / .csv)", "Вставить текст", "Демо-данные"], horizontal=True,
                   label_visibility="collapsed")
    msgs, label = None, ""
    if src.startswith("📄"):
        up = st.file_uploader(".txt: одно сообщение на строку. .csv: колонка с текстом и, по желанию, с датой.",
                              type=["txt", "csv"])
        if up is not None:
            try:
                msgs, label = pick_columns(read_upload(up)), up.name
            except Exception as e:
                st.error(f"Не получилось прочитать файл: {e}")
    elif src.startswith("✍️"):
        pasted = st.text_area("Сообщения, по одному на строку", height=180, key="paste_area")
        lines = [ln.strip() for ln in pasted.splitlines() if ln.strip()]
        if lines:
            msgs, label = pd.DataFrame({"text": lines}), "вставленный текст"
    else:
        msgs, label = make_demo(), "демо-данные"
        st.caption("Две недели «чата поддержки», сгенерированные из шаблонных фраз. На 9-й день после «неудачного обновления» "
                   "негатива становится заметно больше: посмотри, найдёт ли это модель.")

    if msgs is not None:
        msgs = msgs.dropna(subset=["text"]).copy()
        msgs["text"] = msgs["text"].astype(str).str.strip()
        msgs = msgs[msgs["text"] != ""].reset_index(drop=True)
        if msgs.empty:
            st.warning("В файле не нашлось ни одного сообщения.")
        else:
            c1, c2 = st.columns([1, 3])
            limit = c1.number_input("Обработать первые N", 1, max(len(msgs), 1), min(len(msgs), 2000), step=100)
            c2.write("")
            c2.caption(f"Найдено сообщений: {len(msgs)}. Для больших файлов на процессоре начни с нескольких сотен.")
            if st.button("Проанализировать", type="primary"):
                part = msgs.head(int(limit))
                bar = st.progress(0.0, text="Запускаю…")
                df = to_frame(run_model(part["text"].tolist(), model_dir, bar))
                if "date" in part:
                    df["dt"] = to_datetime(part["date"]).reset_index(drop=True)
                bar.empty()
                st.session_state["batch"] = {"df": df, "label": label}

    batch = st.session_state.get("batch")
    if batch:
        st.divider()
        render_dashboard(batch["df"], batch["label"], min_conf)


def render_dashboard(df, label, min_conf):
    n = len(df)
    st.markdown(f"#### Результаты: {label}")
    neg, pos = (df["sentiment"] == "negative").mean(), (df["sentiment"] == "positive").mean()
    mean_score, unsure = df["score"].mean(), int((df["confidence"] < min_conf).sum())
    cols = st.columns(5)
    cols[0].markdown(card("Сообщений", f"{n}", "обработано"), unsafe_allow_html=True)
    cols[1].markdown(card("Негатив", f"{neg:.0%}", f"{int(neg * n)} сообщений", SENT_COLORS["negative"]), unsafe_allow_html=True)
    cols[2].markdown(card("Позитив", f"{pos:.0%}", f"{int(round(pos * n))} сообщений", SENT_COLORS["positive"]), unsafe_allow_html=True)
    cols[3].markdown(card("Средний индекс", f"{mean_score:+.2f}", "от −1 до +1", score_color(mean_score)), unsafe_allow_html=True)
    cols[4].markdown(card("Не уверен", f"{unsure}", f"{unsure / n:.0%} сообщений (порог {min_conf:.2f})"), unsafe_allow_html=True)
    st.write("")

    a, b, c = st.columns(3)
    with a, st.container(border=True):
        show(fig_donut(df), "donut")
    with b, st.container(border=True):
        show(fig_emotion_counts(df), "emo_counts")
    with c, st.container(border=True):
        show(fig_heatmap(df), "heatmap")

    if "dt" in df and df["dt"].notna().sum() > 1:
        st.markdown("##### Динамика")
        choice = st.radio("Период", ["Авто", "Час", "День", "Неделя"], horizontal=True, label_visibility="collapsed")
        unit = auto_unit(df["dt"].dropna()) if choice == "Авто" else choice.lower()
        per = aggregate(df, unit)
        if len(per) < 2:
            st.info(f"Все сообщения попали в один период ({unit}). Выбери период поменьше.")
        else:
            flagged, mu = find_spikes(per)
            fmt = "%d.%m %H:%M" if unit == "час" else "%d.%m.%Y"
            if flagged.any():
                lines = "".join(f"\n- **{ts.strftime(fmt)}**: негатива {row.share_negative:.0%} при обычных {mu:.0%} ({int(row.n)} сообщений)"
                                for ts, row in per[flagged].iterrows())
                st.warning(f"**Всплеск негатива**{lines}")
            elif mu is not None:
                st.success("Резких всплесков негатива не найдено.")
            t1, t2 = st.columns(2)
            with t1, st.container(border=True):
                show(fig_timeline(per, flagged, unit), "timeline")
            with t2, st.container(border=True):
                show(fig_shares(per, unit), "shares")
    elif "dt" in df:
        st.info("Колонку с датой не удалось разобрать, поэтому графика по времени не будет.")

    st.markdown("##### О чём пишут")
    w1, w2 = st.columns(2)
    neg_words = top_words(df.loc[df["sentiment"] == "negative", "text"])
    pos_words = top_words(df.loc[df["sentiment"] == "positive", "text"])
    with w1, st.container(border=True):
        fig = fig_words(neg_words, SENT_COLORS["negative"], "Частые слова в негативных сообщениях")
        if fig:
            show(fig, "words_neg")
        else:
            st.caption("Негативных сообщений нет.")
    with w2, st.container(border=True):
        fig = fig_words(pos_words, SENT_COLORS["positive"], "Частые слова в позитивных сообщениях")
        if fig:
            show(fig, "words_pos")
        else:
            st.caption("Позитивных сообщений нет.")
    st.caption("Слова считаются без лемматизации: «ошибка» и «ошибки» — разные слова.")

    st.markdown("##### Самые яркие сообщения")
    cfg = {"Индекс": st.column_config.NumberColumn(format="%+.2f"), "Сообщение": st.column_config.TextColumn(width="large")}
    uniq = df.drop_duplicates("text")
    t1, t2 = st.columns(2)
    with t1:
        st.caption("Самые негативные")
        st.dataframe(table_view(uniq.nsmallest(8, "score"), min_conf)[["Сообщение", "Индекс"]], hide_index=True, column_config=cfg, **STRETCH)
    with t2:
        st.caption("Самые позитивные")
        st.dataframe(table_view(uniq.nlargest(8, "score"), min_conf)[["Сообщение", "Индекс"]], hide_index=True, column_config=cfg, **STRETCH)

    st.markdown("##### Все сообщения")
    f1, f2, f3, f4 = st.columns([1.3, 1.3, 1.4, 1.2])
    sel_s = f1.multiselect("Тональность", mood.SENT_LABELS, default=mood.SENT_LABELS,
                           format_func=lambda l: f"{SENT_EMOJI[l]} {mood.SENT_RU[l]}")
    sel_e = f2.multiselect("Есть эмоция", mood.EMO_LABELS, format_func=lambda l: f"{EMO_EMOJI[l]} {mood.EMO_RU[l]}")
    query = f3.text_input("Поиск по тексту")
    order = f4.selectbox("Сортировка", ["Как в файле", "Сначала негативные", "Сначала позитивные", "Сначала неуверенные"])
    mask = df["sentiment"].isin(sel_s)
    if sel_e:
        mask &= df[[f"has_{l}" for l in sel_e]].any(axis=1)
    if query:
        mask &= df["text"].str.contains(query, case=False, regex=False)
    view = df[mask]
    if order == "Сначала негативные":
        view = view.sort_values("score")
    elif order == "Сначала позитивные":
        view = view.sort_values("score", ascending=False)
    elif order == "Сначала неуверенные":
        view = view.sort_values("confidence")
    st.caption(f"Показано {len(view)} из {n}")
    st.dataframe(table_view(view, min_conf), hide_index=True, height=min(520, 40 + 35 * max(len(view), 1)), **STRETCH,
                 column_config={"Уверенность": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                                "Индекс": st.column_config.NumberColumn(format="%+.2f"),
                                "Сообщение": st.column_config.TextColumn(width="large"),
                                "Дата": st.column_config.DatetimeColumn(format="DD.MM.YYYY HH:mm")})
    st.download_button("Скачать результаты (CSV)", export_csv(df), "mood_results.csv", "text/csv")


# ----------------------------------------------------------------------------
# Вкладка 3: о модели
# ----------------------------------------------------------------------------
def tab_about(bundle):
    meta = bundle["meta"]
    cols = st.columns(2)
    cols[0].markdown(card("Базовая модель", meta["base_model"].split("/")[-1], "предобученный RuBERT"), unsafe_allow_html=True)
    cols[1].markdown(card("Параметров", f"{bundle['n_params'] / 1e6:.1f} млн", f"макс. длина {meta['max_len']} токенов"), unsafe_allow_html=True)
    #cols[2].markdown(card("Лучшая эпоха", f"{meta.get('epoch', '?')}", "по validation"), unsafe_allow_html=True)
    #cols[3].markdown(card("Устройство", bundle["device"].upper(), "где считается"), unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1, 1.3])
    with left, st.container(border=True):
        st.markdown("##### Пороги эмоций")
        st.dataframe(pd.DataFrame({"Эмоция": [f"{mood.EMO_RU[l]}" for l in mood.EMO_LABELS],
                                   "Порог": meta["thresholds"]}), hide_index=True, **STRETCH,
                     column_config={"Порог": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
        st.caption("Эмоция засчитывается, если её вероятность выше порога. Пороги подобраны на validation-части CEDR.")
    with right, st.container(border=True):
        st.markdown("##### Как читать результаты")
        st.markdown(
            "- **Тональность**: модель выдаёт вероятности трёх классов. *Уверенность* — вероятность лучшего. "
            "Если она ниже порога из боковой панели, интерфейс пишет «не уверен».\n"
            "- **Индекс настроения** = P(позитив) − P(негатив), от −1 до +1.\n"
            "- **Эмоции** независимы: у сообщения может быть несколько эмоций или ни одной.\n")
    with st.expander("Ограничения"):
        st.markdown(
            "- Модель обучена на твитах (RuSentiTweet) и предложениях из блогов и новостей (CEDR). На других текстах точность ниже.\n"
            "- Сарказм и ирония («ну спасибо, очень помогли») ловятся плохо.\n"
            "- Приветствия, поздравления и неясные твиты при обучении исключались, поэтому на них ответ случайнее.\n"
            "- Эмоций всего пять: радость, грусть, удивление, страх, злость. Остальные модель не различает.\n")
    with st.expander("Откуда данные для обучения"):
        st.markdown("- RuSentiTweet: https://huggingface.co/datasets/psytechlab/RuSentiTweet\n"
                    "- CEDR: https://huggingface.co/datasets/sagteam/cedr_v1")


# ----------------------------------------------------------------------------
# Каркас страницы
# ----------------------------------------------------------------------------
st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>Настроение текста</h1>'
            '<p>Тональность и эмоции русского текста</p></div>',
            unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Модель")
    raw_dir = st.text_input("Папка с моделью", "mood_model", help="Там лежат model.pt, meta.json, config.json и файлы токенайзера")
model_dir = resolve_model_dir(raw_dir)

if model_dir is None:
    st.error(f"Не нашёл обученную модель в папке «{raw_dir}».")
    st.markdown("**Что сделать:** скачай `mood_model.zip` из Colab (последняя ячейка блокнота), распакуй в папку "
                "`mood_model` рядом с `app.py` и `mood.py` и обнови страницу:")
    st.code("проект/\n├── app.py\n├── mood.py\n└── mood_model/\n    ├── model.pt\n    ├── meta.json\n"
            "    ├── config.json\n    └── tokenizer.json (и остальные файлы токенайзера)", language="text")
    st.caption("Если папка лежит в другом месте, укажи путь в боковой панели слева.")
    st.stop()

try:
    bundle = load_bundle(model_dir)
except Exception as e:
    st.error(f"Не получилось загрузить модель: {e}")
    st.stop()

with st.sidebar:
    st.success(f"Загружена · {bundle['device'].upper()}")
    min_conf = st.slider("Порог уверенности", 0.40, 0.90, 0.50, 0.01,
                         help="Если уверенность в тональности ниже, интерфейс пишет «не уверен», а не гадает.")
    if st.button("Перезагрузить модель", help="Нажми после замены файлов модели", **STRETCH):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

tab1, tab2 = st.tabs(["Один текст", "О модели"])
with tab1:
    tab_single(model_dir, bundle, min_conf)
with tab2:
    tab_about(bundle)

# tab1, tab2, tab3 = st.tabs(["Один текст", "Пульс чата (много сообщений)", "О модели"])
# with tab1:
#     tab_single(model_dir, bundle, min_conf)
# with tab2:
#     tab_batch(model_dir, min_conf)
# with tab3:
#     tab_about(bundle)
