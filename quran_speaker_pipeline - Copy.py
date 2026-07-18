import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
from yt_dlp import YoutubeDL

COMPETITION_NAME = "Dubai Quran Competition"
QUESTION_WORDS = (
    "السؤال",
    "اقرا من قوله",
    "اقرا من قول الله",
    "من قوله تعالى",
    "قوله تعالى",
    "قال تعالى",
)
STOP_WORDS = (
    "حسب",
    "حسبك",
    "يكفي",
    "قف",
    "توقف",
    "بارك الله",
    "جزاك الله",
)
MERGE_GAP_SECONDS = 1.2
MIN_SPEECH_SECONDS = 0.8


def normalize_arabic(text: str) -> str:
    text = text or ""
    text = re.sub(r"[ًٌٍَُِّْٰـ]", "", text)
    text = (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ة", "ه")
    )
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_video_id(url_or_id: str) -> str:
    value = url_or_id.strip()
    if "youtube.com" not in value and "youtu.be" not in value:
        return re.sub(r"[^A-Za-z0-9_-]", "_", value)
    parsed = urlparse(value)
    if "youtu.be" in parsed.netloc:
        return parsed.path.strip("/").split("/")[0]
    query = parse_qs(parsed.query)
    if query.get("v"):
        return query["v"][0]
    return re.sub(r"[^A-Za-z0-9_-]", "_", value)[-20:]


def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg غير موجود. ثبته أولاً ثم تأكد أن الأمر ffmpeg يعمل داخل PowerShell."
        )


def download_audio(url: str, work_dir: Path) -> tuple[Path, dict[str, Any]]:
    video_id = extract_video_id(url)
    wav_path = work_dir / f"{video_id}.wav"
    raw_template = str(work_dir / f"{video_id}.%(ext)s")
    options = {
        "format": "bestaudio/best",
        "outtmpl": raw_template,
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "postprocessor_args": ["-ac", "1", "-ar", "16000"],
    }
    if wav_path.exists():
        print(f"استخدام ملف الصوت الموجود: {wav_path}")
        with YoutubeDL({"quiet": True, "skip_download": True, "noplaylist": True}) as ydl:
            info = ydl.extract_info(url, download=False)
        return wav_path, info

    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)

    if not wav_path.exists():
        matches = sorted(work_dir.glob(f"{video_id}*.wav"))
        if not matches:
            raise FileNotFoundError("تم التنزيل، لكن لم أجد ملف WAV الناتج.")
        wav_path = matches[0]
    return wav_path, info


def choose_device() -> tuple[str, str]:
    if torch.cuda.is_available():
        return "cuda", "float16"
    return "cpu", "int8"


def transcribe_and_diarize(audio_path: Path, hf_token: str, whisper_model: str,
                            language: str = "ar") -> dict[str, Any]:
    device, compute_type = choose_device()
    batch_size = 16 if device == "cuda" else 4
    print(f"الجهاز المستخدم: {device}")
    print(f"نموذج WhisperX: {whisper_model}")

    model = whisperx.load_model(
        whisper_model, device=device, compute_type=compute_type, language=language
    )
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=batch_size, language=language)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    print("جاري ضبط توقيت الكلمات...")
    align_model, metadata = whisperx.load_align_model(
        language_code=result.get("language", language), device=device
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, device,
        return_char_alignments=False
    )
    del align_model
    if device == "cuda":
        torch.cuda.empty_cache()

    print("جاري فصل صوت السائل عن المتسابق...")
    diarize_model = DiarizationPipeline(
        token=hf_token,
        device=device,
    )
    diarize_segments = diarize_model(
        str(audio_path),
        min_speakers=2,
        max_speakers=3,
    )
    return assign_word_speakers(
        diarize_segments,
        result,
        fill_nearest=True,
    )


def segment_speaker(segment: dict[str, Any]) -> str:
    speaker = segment.get("speaker")
    if speaker:
        return str(speaker)
    counts: dict[str, int] = {}
    for word in segment.get("words", []):
        value = word.get("speaker")
        if value:
            counts[str(value)] = counts.get(str(value), 0) + 1
    return max(counts, key=counts.get) if counts else "UNKNOWN"


def merge_same_speaker_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not segments:
        return []
    merged = [segments[0].copy()]
    for current in segments[1:]:
        previous = merged[-1]
        gap = current["start"] - previous["end"]
        if current["speaker"] == previous["speaker"] and gap <= MERGE_GAP_SECONDS:
            previous["end"] = max(previous["end"], current["end"])
            previous["text"] = f'{previous["text"]} {current["text"]}'.strip()
            previous["norm"] = normalize_arabic(previous["text"])
            previous.setdefault("words", []).extend(current.get("words", []))
        else:
            merged.append(current.copy())
    return merged


def clean_segments(result: dict[str, Any]) -> list[dict[str, Any]]:
    cleaned = []
    for segment in result.get("segments", []):
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        text = (segment.get("text") or "").strip()
        if end - start < MIN_SPEECH_SECONDS and not text:
            continue
        words = []
        for word in segment.get("words", []):
            word_text = str(word.get("word", "")).strip()
            if not word_text:
                continue
            words.append({
                "word": word_text,
                "start": float(word.get("start", start)),
                "end": float(word.get("end", word.get("start", start))),
                "speaker": str(word.get("speaker", segment_speaker(segment))),
            })

        cleaned.append({
            "start": start,
            "end": end,
            "text": text,
            "norm": normalize_arabic(text),
            "speaker": segment_speaker(segment),
            "words": words,
        })
    return merge_same_speaker_segments(cleaned)


def contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    norm = normalize_arabic(text)
    return any(normalize_arabic(p) in norm for p in phrases)


def looks_like_question(text: str) -> bool:
    norm = normalize_arabic(text)

    question_forms = ("السؤال", "سؤال", "سوال", "اسؤال")
    instruction_forms = (
        "اقرا من قوله",
        "اقرا من قول الله",
        "من قوله تعالى",
        "قوله تعالى",
        "قال تعالى",
    )

    return (
        any(normalize_arabic(x) in norm for x in question_forms)
        or any(normalize_arabic(x) in norm for x in instruction_forms)
    )


def _word_matches_question(word: str) -> bool:
    norm = normalize_arabic(word)
    return any(
        form in norm
        for form in ("السؤال", "سؤال", "سوال", "اسؤال")
    )


def _word_matches_stop(word: str) -> bool:
    norm = normalize_arabic(word)
    return any(normalize_arabic(stop) in norm for stop in STOP_WORDS)


def estimate_phrase_time(
    segment: dict[str, Any],
    phrase_kind: str,
) -> float:
    """
    تحديد وقت السؤال أو كلمة الإيقاف داخل المقطع المختلط.

    - للسؤال: نأخذ أول ظهور لكلمة السؤال.
    - للإيقاف: نأخذ آخر كلمة إيقاف قبل بداية السؤال التالي داخل
      المقطع نفسه، لأن رئيس اللجنة قد يقول: حسب... حسبك.
    """
    words = segment.get("words", [])

    if words:
        if phrase_kind == "question":
            for word in words:
                value = str(word.get("word", ""))
                if _word_matches_question(value):
                    return float(word.get("start", segment["start"]))

        if phrase_kind == "stop":
            last_stop_time = None
            question_time = None

            for word in words:
                value = str(word.get("word", ""))
                word_start = float(word.get("start", segment["start"]))

                if question_time is None and _word_matches_question(value):
                    question_time = word_start

                if _word_matches_stop(value):
                    if question_time is None or word_start < question_time:
                        last_stop_time = word_start

            if last_stop_time is not None:
                return last_stop_time

    text_value = segment.get("text", "")
    norm = normalize_arabic(text_value)

    candidates = (
        ("السؤال", "سؤال", "سوال", "اسؤال")
        if phrase_kind == "question"
        else tuple(normalize_arabic(x) for x in STOP_WORDS)
    )

    positions = [norm.find(item) for item in candidates if norm.find(item) >= 0]
    if not positions or not norm:
        return float(segment["start"])

    position = min(positions) if phrase_kind == "question" else max(positions)
    ratio = position / max(1, len(norm))
    duration = float(segment["end"]) - float(segment["start"])
    return float(segment["start"]) + ratio * duration


def question_start_time(segment: dict[str, Any]) -> float:
    return estimate_phrase_time(segment, "question")


def stop_time_in_segment(segment: dict[str, Any]) -> float | None:
    if not contains_any(segment.get("text", ""), STOP_WORDS):
        return None
    return estimate_phrase_time(segment, "stop")


def find_question_anchor_indices(
    segments: list[dict[str, Any]],
) -> list[int]:
    """
    تحديد المقطع الذي يحتوي السؤال نفسه، لا المقطع السابق له.

    في النسخة السابقة كنا نفحص نافذة من مقطعين ونضيف index الحالي،
    لذلك كان السؤال الثاني يبدأ عند 215 بدلاً من 243، والثالث عند
    455 بدلاً من 470. هنا نختار المقطع الفعلي الذي يحتوي كلمة السؤال.
    """
    candidates: list[int] = []

    for index, segment in enumerate(segments):
        # الحالة الطبيعية: السؤال موجود داخل المقطع نفسه.
        if looks_like_question(segment["text"]):
            candidates.append(index)
            continue

        # الحالة النادرة: العبارة مقسمة بين مقطعين متتاليين.
        if index + 1 < len(segments):
            combined = f'{segment["text"]} {segments[index + 1]["text"]}'
            if looks_like_question(combined):
                # إذا كان المقطع التالي يحتوي كلمة السؤال أو عبارة اقرأ،
                # فهو بداية السؤال الحقيقية، وليس المقطع الحالي.
                if looks_like_question(segments[index + 1]["text"]):
                    candidates.append(index + 1)
                else:
                    candidates.append(index)

    # إزالة التكرار مع الحفاظ على الترتيب.
    candidates = list(dict.fromkeys(candidates))

    anchors: list[int] = []
    for index in candidates:
        if not anchors:
            anchors.append(index)
            continue

        previous_time = segments[anchors[-1]]["start"]
        current_time = segments[index]["start"]

        # يمنع احتساب السؤال نفسه أكثر من مرة.
        if current_time - previous_time >= 35:
            anchors.append(index)

    return anchors


def find_answer_start_index(
    segments: list[dict[str, Any]],
    anchor_index: int,
    next_anchor_index: int,
) -> int | None:
    anchor_speaker = segments[anchor_index]["speaker"]

    for index in range(anchor_index + 1, next_anchor_index):
        segment = segments[index]
        duration = segment["end"] - segment["start"]

        if duration < MIN_SPEECH_SECONDS:
            continue
        if looks_like_question(segment["text"]):
            continue
        if segment["speaker"] not in ("UNKNOWN", anchor_speaker):
            return index

    for index in range(anchor_index + 1, next_anchor_index):
        segment = segments[index]
        duration = segment["end"] - segment["start"]

        if duration < MIN_SPEECH_SECONDS:
            continue
        if looks_like_question(segment["text"]):
            continue

        return index

    return None


def find_answer_end_hybrid(
    segments: list[dict[str, Any]],
    answer_index: int,
    next_anchor_index: int,
    next_question_start: float | None,
) -> float:
    """
    نهاية الجواب تكون عند أول مقطع يحتوي كلمة إيقاف حقيقية بعد
    بدء المتسابق، مع إعطاء الأولوية لآخر "حسب/حسبك" قبل السؤال التالي
    إذا حدث التداخل داخل المقطع نفسه.
    """
    last_end = segments[answer_index]["end"]
    best_stop_time = None

    for index in range(answer_index, min(next_anchor_index + 1, len(segments))):
        segment = segments[index]
        last_end = max(last_end, segment["end"])

        stop_time = stop_time_in_segment(segment)
        if stop_time is not None:
            if next_question_start is None or stop_time < next_question_start:
                best_stop_time = stop_time

        if index == next_anchor_index:
            break

    if best_stop_time is not None:
        return best_stop_time

    if next_question_start is not None:
        return next_question_start

    return last_end

def build_questions(
    segments: list[dict[str, Any]],
) -> list[dict[str, int]]:
    anchors = find_question_anchor_indices(segments)

    if not anchors:
        print("لم أعثر على أسئلة في النص الناتج.")
        return []

    questions: list[dict[str, int]] = []

    for order, anchor_index in enumerate(anchors, start=1):
        anchor = segments[anchor_index]
        current_question_start = question_start_time(anchor)

        next_anchor_index = (
            anchors[order]
            if order < len(anchors)
            else len(segments)
        )

        next_question_start = None
        if next_anchor_index < len(segments):
            next_question_start = question_start_time(
                segments[next_anchor_index]
            )

        answer_index = find_answer_start_index(
            segments,
            anchor_index,
            next_anchor_index,
        )

        if answer_index is None:
            print(
                f"تحذير: لم أجد بداية جواب واضحة بعد السؤال {order}."
            )
            continue

        answer = segments[answer_index]
        answer_start_value = float(answer["start"])

        answer_end = find_answer_end_hybrid(
            segments,
            answer_index,
            next_anchor_index,
            next_question_start,
        )

        # لا نسمح لنهاية الجواب بتجاوز بداية السؤال التالي.
        if next_question_start is not None and answer_end >= next_question_start:
            answer_end = max(answer_start_value, next_question_start - 0.05)

        question_start = round(current_question_start)
        answer_start = round(answer_start_value)
        question_end = max(question_start, answer_start - 1)

        questions.append({
            "id": len(questions) + 1,
            "questionStart": max(0, question_start),
            "questionEnd": question_end,
            "answerStart": max(question_end + 1, answer_start),
            "answerEnd": max(answer_start, round(answer_end)),
        })

    return questions

def save_debug_segments(path: Path, segments: list[dict[str, Any]]) -> None:
    payload = [{
        "start": round(s["start"], 2),
        "end": round(s["end"], 2),
        "speaker": s["speaker"],
        "text": s["text"],
        "words": s.get("words", []),
    } for s in segments]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")



def load_debug_segments(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    segments: list[dict[str, Any]] = []

    for item in raw:
        segment_text = str(item.get("text", "")).strip()
        segments.append({
            "start": float(item.get("start", 0.0)),
            "end": float(item.get("end", 0.0)),
            "speaker": str(item.get("speaker", "UNKNOWN")),
            "text": segment_text,
            "norm": normalize_arabic(segment_text),
            "words": item.get("words", []),
        })

    return segments

def process_video(
    url: str,
    output_dir: Path,
    work_dir: Path,
    hf_token: str,
    whisper_model: str,
) -> Path:
    video_id = extract_video_id(url)

    print("\n" + "=" * 70)
    print(f"معالجة الفيديو: {video_id}")

    debug_path = output_dir / f"{video_id}_speakers.json"

    if debug_path.exists():
        print(f"استخدام ملف المتحدثين الموجود: {debug_path}")
        segments = load_debug_segments(debug_path)

        with YoutubeDL({
            "quiet": True,
            "skip_download": True,
            "noplaylist": True,
        }) as ydl:
            info = ydl.extract_info(url, download=False)
    else:
        audio_path, info = download_audio(url, work_dir)
        result = transcribe_and_diarize(
            audio_path,
            hf_token,
            whisper_model,
        )
        segments = clean_segments(result)
        save_debug_segments(debug_path, segments)

    questions = build_questions(segments)

    output = {
        "competition": COMPETITION_NAME,
        "videoId": video_id,
        "questions": questions,
    }

    output_path = output_dir / f"{video_id}.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"العنوان: {info.get('title', '')}")
    print(f"عدد الأسئلة: {len(questions)}")
    print(f"تم إنشاء: {output_path}")
    print(f"ملف التدقيق: {debug_path}")

    return output_path

def load_urls(args: argparse.Namespace) -> list[str]:
    urls = []
    if args.url:
        urls.append(args.url.strip())
    if args.urls_file:
        file_path = Path(args.urls_file)
        if not file_path.exists():
            raise FileNotFoundError(f"ملف الروابط غير موجود: {file_path}")
        for line in file_path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return list(dict.fromkeys(urls))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="تنزيل فيديوهات مسابقات القرآن وفصل المتحدثين واستخراج أزمنة الأسئلة."
    )
    parser.add_argument("--url", help="رابط فيديو YouTube واحد")
    parser.add_argument("--urls-file", help="ملف TXT يحتوي رابطاً واحداً في كل سطر")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--work-dir", default="audio_cache")
    parser.add_argument(
        "--model", default="small",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        help="استخدم small أو medium على جهاز بلا GPU."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_ffmpeg()

    hf_token = os.getenv("HF_TOKEN", "").strip()
    if not hf_token:
        print(
            "خطأ: ضع Hugging Face token في متغير البيئة HF_TOKEN.\n"
            'PowerShell:\n$env:HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"\n'
        )
        sys.exit(1)

    urls = load_urls(args)
    if not urls:
        entered = input("ضع رابط YouTube، أو اتركه فارغاً لاستخدام links.txt: ").strip()
        if entered:
            urls = [entered]
        elif Path("links.txt").exists():
            args.urls_file = "links.txt"
            urls = load_urls(args)

    if not urls:
        print("لم يتم توفير أي رابط.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    for url in urls:
        try:
            process_video(url, output_dir, work_dir, hf_token, args.model)
        except Exception as exc:
            print(f"فشل الفيديو:\n{url}\nالسبب: {exc}")
            failures.append({"url": url, "error": str(exc)})

    if failures:
        failure_path = output_dir / "failures.json"
        failure_path.write_text(
            json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nتم تسجيل الأخطاء في: {failure_path}")

    print("\nانتهت المعالجة.")


if __name__ == "__main__":
    main()
