import json
import re
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi


COMPETITION_NAME = "Dubai Quran Competition"
OUTPUT_FILE = "sample_competition.json"

# مدة مقطع السؤال إذا لم نستطع تحديد بداية الإجابة من الترجمة
DEFAULT_QUESTION_DURATION = 15

# عدد الأسطر التي ندمجها معًا لأن YouTube يقطع الجملة أحيانًا:
# مثال: "اقرا من قوله" في سطر و "تعالى" في السطر التالي.
LOOKAHEAD_LINES = 5

STOP_WORDS = ["حسبك", "يكفي", "تمام"]

QUESTION_WORDS = [
    "السؤال",
    "السوال",   # خطأ شائع في transcript
]

READ_PHRASES = [
    "اقرا من قوله تعالى",
    "اقرا من قول الله تعالى",
    "اقرا من قوله",
    "من قوله تعالى",
    "قوله تعالى",
    "قال تعالى",
]

SKIP_ANSWER_WORDS = {
    "السؤال", "السوال", "الاول", "الأول", "الثاني", "الثالث", "الرابع", "الخامس",
    "اقرا", "من", "قوله", "تعالى", "تعالي", "قال", "الله", "سبحانه",
    "الاية", "الايات", "وابدا", "ابدا", "عندي",
}

ORDINAL_WORDS = [
    "الاول", "الثاني", "الثالث", "الرابع", "الخامس", "السادس", "السابع",
    "الثامن", "التاسع", "العاشر", "الحادي", "الحادي عشر", "الثاني عشر",
]


def extract_video_id(url_or_id: str) -> str:
    text = url_or_id.strip()

    if "youtube.com" not in text and "youtu.be" not in text:
        return text

    parsed = urlparse(text)

    if "youtu.be" in parsed.netloc:
        return parsed.path.strip("/")

    query = parse_qs(parsed.query)
    if "v" in query:
        return query["v"][0]

    raise ValueError("لم أستطع استخراج videoId من الرابط")


def normalize_arabic(text: str) -> str:
    text = str(text)
    text = re.sub(r"[ًٌٍَُِّْٰـ]", "", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")
    text = text.replace("ؤ", "و").replace("ئ", "ي")
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_transcript(video_id: str):
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

        if not text:
            continue

        rows.append({
            "time": start,
            "text": text,
            "norm": normalize_arabic(text),
        })

    return rows


def window_text(rows, index, count=LOOKAHEAD_LINES):
    selected = rows[index:index + count]
    text = " ".join(row["text"] for row in selected)
    norm = " ".join(row["norm"] for row in selected)
    return text, norm


def has_question_word(norm: str) -> bool:
    words = set(norm.split())
    return any(normalize_arabic(w) in words for w in QUESTION_WORDS)


def has_read_phrase(norm: str) -> bool:
    return any(normalize_arabic(p) in norm for p in READ_PHRASES)


def is_probable_question_start(rows, index) -> bool:
    """
    نعتبره سؤالًا فقط إذا وُجدت كلمة السؤال في نافذة الأسطر، ومعها عبارة قراءة.
    هذا يمنع اعتبار المقدمة أو الآيات سؤالًا وهميًا.
    """
    _, norm = window_text(rows, index)

    if not has_question_word(norm):
        return False

    # غالبًا السؤال الحقيقي يحتوي أيضًا عبارة القراءة خلال الأسطر القريبة.
    if has_read_phrase(norm):
        return True

    # أحيانًا transcript يكتب: "السؤال الثاني" ثم بعده مباشرة الآية بدون عبارة اقرأ.
    # نقبلها إذا ظهر رقم/ترتيب السؤال، لكن لا نقبل كلمة "سؤال" وحدها.
    if any(normalize_arabic(w) in norm for w in ORDINAL_WORDS):
        return True

    return False


def get_question_starts(rows):
    starts = []
    last_time = -999

    for i in range(len(rows)):
        if not is_probable_question_start(rows, i):
            continue

        start_time = rows[i]["time"]

        # منع تكرار نفس السؤال بسبب الأسطر المتقاربة: السؤال / اقرأ / تعالى...
        if start_time - last_time < 20:
            continue

        starts.append({
            "index": i,
            "time": start_time,
            "text": window_text(rows, i)[0],
            "norm": window_text(rows, i)[1],
        })
        last_time = start_time

    return starts


def first_meaningful_answer_word(question_norm: str):
    """
    يحاول استخراج أول كلمة من الآية بعد عبارة السؤال.
    مثال: السؤال الاول اقرا من قوله تعالى يا ايها النبي...
    يرجع: يا
    """
    norm = question_norm

    # نحذف كل شيء قبل آخر ظهور لكلمة تعالى أو عبارة قال تعالى إن وجدت.
    markers = ["تعالى", "تعالي", "قال"]
    last_pos = -1
    for marker in markers:
        pos = norm.rfind(normalize_arabic(marker))
        if pos > last_pos:
            last_pos = pos + len(normalize_arabic(marker))

    clue = norm[last_pos:].strip() if last_pos != -1 else norm
    words = clue.split()

    for word in words:
        if word not in SKIP_ANSWER_WORDS and len(word) > 1:
            return word

    return ""


def find_word_after(rows, word, after_time, before_time=None):
    word = normalize_arabic(word)
    if not word:
        return None

    for row in rows:
        if row["time"] <= after_time:
            continue
        if before_time is not None and row["time"] >= before_time:
            return None

        words = row["norm"].split()
        if word in words:
            return row["time"]

    return None


def find_stop_after(rows, after_time, before_time=None):
    stops = {normalize_arabic(w) for w in STOP_WORDS}

    for row in rows:
        if row["time"] <= after_time:
            continue

        if before_time is not None and row["time"] >= before_time:
            return max(after_time, before_time - 1)

        words = set(row["norm"].split())
        if words.intersection(stops):
            return row["time"]

    if before_time is not None:
        return max(after_time, before_time - 1)

    return after_time + 60


def build_questions(rows):
    starts = get_question_starts(rows)

    if not starts:
        print("لم أجد أي سؤال.")
        print("تأكد أن transcript يحتوي كلمة: السؤال، أو عبارة: اقرأ من قوله تعالى")
        return []

    questions = []

    for i, q in enumerate(starts):
        question_id = i + 1
        question_start = q["time"]

        next_question_start = starts[i + 1]["time"] if i + 1 < len(starts) else None

        first_word = first_meaningful_answer_word(q["norm"])

        # بداية السؤال تنتهي عند ظهور أول كلمة من التلاوة مرة أخرى بعد نص السؤال.
        answer_start = find_word_after(
            rows,
            first_word,
            after_time=question_start,
            before_time=next_question_start,
        )

        # إذا لم نجد كلمة الآية، نستخدم مدة افتراضية للسؤال.
        if answer_start is None:
            question_end = question_start + DEFAULT_QUESTION_DURATION
            answer_start = question_end + 1
        else:
            question_end = max(question_start, answer_start - 2)

        answer_end = find_stop_after(
            rows,
            after_time=answer_start,
            before_time=next_question_start,
        )

        questions.append({
            "id": question_id,
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
    print("\n--- أول 80 سطر من transcript ---")
    for row in rows[:80]:
        print(row["time"], "=>", row["text"])
    print("--- انتهى العرض ---\n")

    questions = build_questions(rows)

    data = {
        "competition": COMPETITION_NAME,
        "videoId": video_id,
        "questions": questions,
    }

    Path(OUTPUT_FILE).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"\nتم إنشاء الملف: {OUTPUT_FILE}")
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
