import json
import tkinter as tk
from tkinter import messagebox

questions = []
question_id = 1

def to_seconds(t):
    parts = t.strip().split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return int(t)

def save_question():
    global question_id

    try:
        q_start = to_seconds(q_start_entry.get())
        q_end = to_seconds(q_end_entry.get())
        a_start = to_seconds(a_start_entry.get())
        a_end = to_seconds(a_end_entry.get())
    except:
        messagebox.showerror("خطأ", "اكتب الوقت مثل 1:15")
        return

    questions.append({
        "id": question_id,
        "questionStart": q_start,
        "questionEnd": q_end,
        "answerStart": a_start,
        "answerEnd": a_end
    })

    question_id += 1
    question_label.config(text=f"السؤال رقم {question_id}")

    q_start_entry.delete(0, tk.END)
    q_end_entry.delete(0, tk.END)
    a_start_entry.delete(0, tk.END)
    a_end_entry.delete(0, tk.END)

    messagebox.showinfo("تم", "تم حفظ السؤال")

def export_json():
    data = {
        "competition": competition_entry.get().strip(),
        "videoId": video_id_entry.get().strip(),
        "questions": questions
    }

    with open("sample_competition.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    messagebox.showinfo("تم", "تم إنشاء sample_competition.json")

root = tk.Tk()
root.title("JSON Maker")
root.geometry("420x520")

tk.Label(root, text="اسم المسابقة").pack(pady=5)
competition_entry = tk.Entry(root, width=40)
competition_entry.insert(0, "Dubai Quran Competition")
competition_entry.pack()

tk.Label(root, text="YouTube Video ID").pack(pady=5)
video_id_entry = tk.Entry(root, width=40)
video_id_entry.insert(0, "y5yd7b2AZnE")
video_id_entry.pack()

question_label = tk.Label(root, text="السؤال رقم 1", font=("Arial", 16, "bold"))
question_label.pack(pady=15)

tk.Label(root, text="بداية السؤال مثل 0:46").pack()
q_start_entry = tk.Entry(root, width=30)
q_start_entry.pack(pady=5)

tk.Label(root, text="نهاية السؤال مثل 1:10").pack()
q_end_entry = tk.Entry(root, width=30)
q_end_entry.pack(pady=5)

tk.Label(root, text="بداية الإجابة مثل 1:10").pack()
a_start_entry = tk.Entry(root, width=30)
a_start_entry.pack(pady=5)

tk.Label(root, text="نهاية الإجابة مثل 2:00").pack()
a_end_entry = tk.Entry(root, width=30)
a_end_entry.pack(pady=5)

tk.Button(root, text="حفظ السؤال", command=save_question, bg="blue", fg="white", width=25).pack(pady=20)
tk.Button(root, text="تصدير JSON", command=export_json, bg="purple", fg="white", width=25).pack(pady=5)

root.mainloop()