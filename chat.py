import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertConfig
import tkinter as tk
from tkinter import scrolledtext

pretrained_model_path = "model"         
scratch_model_path = "model_scratch"    
data_path = "E:/aiproject/data.json"    

# ---------------- Load JSON ----------------
with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# ---------------- Prepare label -> answers dictionary ----------------
label_to_answers = {}
for item in data:
    label = item["label"]
    answer = item.get("answer", item["question"])
    if label not in label_to_answers:
        label_to_answers[label] = []
    label_to_answers[label].append(answer)

# ---------------- Encode labels ----------------
unique_labels = sorted(set([item["label"] for item in data]))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

# ---------------- Load Tokenizer ----------------
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

# ---------------- Load Pretrained Model ----------------
model_pretrained = AutoModelForSequenceClassification.from_pretrained(pretrained_model_path)
model_pretrained.eval()

# ---------------- Create Model from Scratch ----------------
config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
)
model_scratch = AutoModelForSequenceClassification.from_config(config)
model_scratch.eval()

# ---------------- Function to Get Answer ----------------
def get_answer(question, use_scratch=False):
    """
    use_scratch: True → use model from scratch
                 False → use pretrained model
    """
    m = model_scratch if use_scratch else model_pretrained
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = m(**inputs)
    logits = outputs.logits
    predicted_label_id = logits.argmax().item()
    label = m.config.id2label[predicted_label_id]
    
    if label in label_to_answers:
        return random.choice(label_to_answers[label])
    else:
        return "Sorry, I don't know the answer."

# ---------------- GUI ----------------
window = tk.Tk()
window.title("AI Chatbot")
window.geometry("700x500")

# Chat area
chat_area = scrolledtext.ScrolledText(window, wrap=tk.WORD, font=("Arial", 12))
chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
chat_area.configure(state='disabled')

# Entry frame
entry_frame = tk.Frame(window, height=50)
entry_frame.pack(fill=tk.X, padx=10, pady=5, side=tk.BOTTOM)

# Entry box
entry = tk.Entry(entry_frame, font=("Arial", 14))
entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5), pady=5)

# Checkbox to select scratch model
use_scratch_var = tk.BooleanVar()
scratch_check = tk.Checkbutton(entry_frame, text="Use Model from Scratch", variable=use_scratch_var)
scratch_check.pack(side=tk.LEFT, padx=5)

# Send button function
def send():
    question = entry.get()
    if question.strip() == "":
        return
    chat_area.configure(state='normal')
    chat_area.insert(tk.END, "You: " + question + "\n")
    answer = get_answer(question, use_scratch=use_scratch_var.get())
    chat_area.insert(tk.END, "Bot: " + answer + "\n\n")
    chat_area.configure(state='disabled')
    chat_area.yview(tk.END)
    entry.delete(0, tk.END)

# Send button
send_button = tk.Button(entry_frame, text="Send", width=10, font=("Arial", 12), command=send)
send_button.pack(side=tk.RIGHT, padx=5, pady=5)

# Allow Enter key to send
entry.bind("<Return>", lambda event: send())

# Run GUI
window.mainloop()
