import json
import re
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi


COMPETITION_NAME = "Dubai Quran Competition"
OUTPUT_FILE = "sample_competition.json"

MIN_FIRST_QUESTION_TIME = 35
STOP_WORDS = ["حسبك", "يكفي", "قف"]

QUESTION_CLUES = [
    "اقرا من قوله",
    "اقرا من قول الله",
    "من قوله تعالى",
    "قوله تعالى",
    "قال تعالى",
    "اقرا",
]


def extract_video_id(url_or_id):
    text = url_or_id.strip()

    if "youtube.com" not in text and "youtu.be" not in text:
        return text

    parsed = urlparse(text)

    if "youtu.be" in parsed.netloc:
        return parsed.path.strip("/").split("?")[0]

    query = parse_qs(parsed.query)
    if "v" in query:
        return query["v"][0]

    raise ValueError("لم أستطع استخراج videoId")


def normalize_arabic(text):
    text = re.sub(r"[ًٌٍَُِّْٰـ]", "", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_transcript(video_id):
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

    try:
        transcript = transcript_list.find_transcript(["ar"])
    except Exception:
        try:
            transcript = transcript_list.find_generated_transcript(["ar"])
        except Exception:
            transcript = transcript_list.find_transcript(["en"])

    data = transcript.fetch()
    rows = []

    for item in data:
        if isinstance(item, dict):
            start = int(float(item["start"]))
            text = item["text"].replace("\n", " ").strip()
        else:
            start = int(float(item.start))
            text = item.text.replace("\n", " ").strip()

        if text:
            rows.append({
                "time": start,
                "text": text,
                "norm": normalize_arabic(text),
            })

    return rows


def is_question_window(text):
    norm = normalize_arabic(text)

    if "السؤال" not in norm:
        return False

    return any(clue in norm for clue in QUESTION_CLUES)


def find_question_starts(rows, window_size=8):
    starts = []
    used_times = []

    for i in range(len(rows)):
        if rows[i]["time"] < MIN_FIRST_QUESTION_TIME:
            continue

        window = rows[i:i + window_size]
        window_text = " ".join(r["text"] for r in window)

        if not is_question_window(window_text):
            continue

        start_row = None
        for r in window:
            if "السؤال" in r["norm"]:
                start_row = r
                break

        if start_row is None:
            continue

        if any(abs(start_row["time"] - t) < 60 for t in used_times):
            continue

        starts.append({
            "time": start_row["time"],
            "text": window_text,
            "norm": normalize_arabic(window_text),
        })

        used_times.append(start_row["time"])

    starts.sort(key=lambda x: x["time"])
    return starts


def get_clue_words(question_text):
    norm = normalize_arabic(question_text)

    remove_words = {
        "السؤال", "الاول", "الثاني", "الثالث", "الرابع", "الخامس",
        "السادس", "السابع", "الثامن", "التاسع", "العاشر",
        "اقرا", "من", "قوله", "قول", "الله", "تعالى", "قال",
        "اعوذ", "بالله", "الشيطان", "الرجيم",
        "بسم", "الرحمن", "الرحيم",
        "حسبك", "يكفي", "قف",
    }

    words = []
    for w in norm.split():
        if w not in remove_words and len(w) > 1:
            words.append(w)

    return words[:5]


def row_contains_clue(row_norm, clue_words):
    if not clue_words:
        return False

    row_words = row_norm.split()
    count = 0

    for w in clue_words:
        if w in row_words:
            count += 1

    return count >= min(2, len(clue_words))


def find_answer_start_by_repetition(rows, question_start, question_text, next_question_start=None):
    clue_words = get_clue_words(question_text)

    seen_count = 0

    for r in rows:
        if r["time"] <= question_start:
            continue

        if next_question_start is not None and r["time"] >= next_question_start:
            break

        if row_contains_clue(r["norm"], clue_words):
            seen_count += 1

            if seen_count >= 2:
                return r["time"]

    return question_start + 20


def find_question_end(rows, question_start, answer_start):
    end = answer_start - 1

    for r in rows:
        if r["time"] <= question_start:
            continue

        if r["time"] >= answer_start:
            break

        end = r["time"]

    return max(question_start + 2, end)


def find_stop_after(rows, answer_start, next_question_start=None):
    stop_words = [normalize_arabic(w) for w in STOP_WORDS]

    for r in rows:
        if r["time"] <= answer_start:
            continue

        if next_question_start is not None and r["time"] >= next_question_start:
            return max(answer_start + 1, next_question_start - 1)

        words = r["norm"].split()

        if any(stop in words for stop in stop_words):
            return r["time"]

    if next_question_start is not None:
        return max(answer_start + 1, next_question_start - 1)

    return answer_start + 180


def build_questions(rows):
    starts = find_question_starts(rows)

    if not starts:
        print("لم أجد أي سؤال.")
        return []

    questions = []

    for i, q in enumerate(starts):
        question_start = q["time"]

        next_question_start = None
        if i + 1 < len(starts):
            next_question_start = starts[i + 1]["time"]

        answer_start = find_answer_start_by_repetition(
            rows=rows,
            question_start=question_start,
            question_text=q["text"],
            next_question_start=next_question_start,
        )

        question_end = find_question_end(
            rows=rows,
            question_start=question_start,
            answer_start=answer_start,
        )

        answer_end = find_stop_after(
            rows=rows,
            answer_start=answer_start,
            next_question_start=next_question_start,
        )

        questions.append({
            "id": i + 1,
            "questionStart": int(question_start),
            "questionEnd": int(question_end),
            "answerStart": int(answer_start),
            "answerEnd": int(answer_end),
        })

    return questions


def main():
    url_or_id = input("ضع رابط YouTube أو videoId: ").strip()
    video_id = extract_video_id(url_or_id)

    print("جاري جلب transcript من YouTube...")
    rows = fetch_transcript(video_id)

    print(f"تم جلب {len(rows)} سطر من transcript.")

    questions = build_questions(rows)

    data = {
        "competition": COMPETITION_NAME,
        "videoId": video_id,
        "questions": questions,
    }

    Path(OUTPUT_FILE).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nتم إنشاء الملف: {OUTPUT_FILE}")
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()