from flask import Flask, request
import requests
from PIL import Image
from io import BytesIO
import mimetypes
import base64
import requests
from collections import defaultdict
from openai import OpenAI
from gtts import gTTS
from pydub import AudioSegment
import os

app = Flask(__name__)

# Configuration
VERIFY_TOKEN = "verifyme123"
ACCESS_TOKEN = "EAATCdCwA7KIBPHd4K7y1qM9A9YbbPyZBWQ72fi2poqZB8AB6qvlgMZA1RhQWQTxBDCviVMi6rkWZAak7mnTZCj0kZAHOQX21AEM7gN7I2ZApCZCDatay8kSJorb6WI0jZBZB7QJAQqZCQDFKrq77GESMAp94zj13NlWjpzFCSRx7Fh6pPXVUuW4KjBvD8ztZATeBX4ZALg4CBzMBVSLQoX1HN5qleQ8v1FZACzjZAiuzNAUfhmXgztlyokZD"
PHONE_NUMBER_ID = "699791233214714"
OPENAI_API_KEY = "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"

client = OpenAI(api_key=OPENAI_API_KEY)
user_memory = defaultdict(list)
MAX_HISTORY = 20

# === Domain-specific rubric for OS (Operating Systems) ===
OS_SYLLABUS_CONTEXT = """
Syllabus for Operating Systems includes:
- Process Management
- Memory Management
- File Systems
- CPU Scheduling Algorithms
- Deadlock
- Virtual Memory
- I/O Systems
- System Calls
- Threads and Concurrency
- Paging and Segmentation
- Linux Architecture Basics
Textbook reference: Modern Operating Systems by Andrew S. Tanenbaum
Notes: Provided lecture notes cover the above topics only.
"""

# === Webhook Verification ===
@app.route("/webhook", methods=["GET"])
def verify():
    if request.args.get("hub.verify_token") == VERIFY_TOKEN:
        return request.args.get("hub.challenge")
    return "Verification token mismatch", 403

# === POST handler ===
@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    try:
        for entry in data.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                messages = value.get("messages")
                if messages:
                    msg = messages[0]
                    sender = msg["from"]
                    if msg.get("type") == "text":
                        text = msg["text"]["body"]
                        task = extract_task_info(text)

                        subject = task["subject"]
                        topic = task["topic"]
                        agent = task["agent"]

                        print(f"Routing to agent: {agent}, Subject: {subject}, Topic: {topic}")

                        if subject.lower() == "os" or subject.lower() == "operating system":
                            if agent == "WorksheetGeneratorAgent":
                                reply = worksheet_generator_os(topic)
                            elif agent == "LocalizedContentAgent":
                                reply = localized_content_os(topic)
                            else:
                                reply = f"Agent: {agent}\nSubject: {subject}\nTopic: {topic}"
                        else:
                            reply = f"Only OS subject is currently supported."

                        send_whatsapp_message(sender, reply)
    except Exception as e:
        print("Error:", e)
    return "ok", 200

# === Task Extractor ===
def extract_task_info(message):
    prompt = f"""
Extract the task from the teacher's input.
Return a JSON object with subject, topic, and agent name (from: LocalizedContentAgent, WorksheetGeneratorAgent, VisualAidGeneratorAgent, StudentDoubtSolverAgent, LessonPlannerAgent)

Input: {message}

Respond in JSON:
{{
  "subject": "",
  "topic": "",
  "agent": ""
}}
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You classify teacher inputs."},
            {"role": "user", "content": prompt}
        ]
    )
    return json.loads(response.choices[0].message.content.strip())

# === OS-Specific Agents with Context Guard ===
def worksheet_generator_os(topic):
    prompt = f"""
You are the Worksheet Generator Agent for Operating Systems.
Only answer if the topic '{topic}' is within the following syllabus:
{OS_SYLLABUS_CONTEXT}

If the topic is not in the syllabus, say "The topic '{topic}' is out of syllabus."

Otherwise, generate a worksheet with:
- 2 MCQs
- 2 short answer questions
- 1 descriptive question

Be curriculum-aligned and relevant.
"""
    return call_gpt(prompt)

def localized_content_os(topic):
    prompt = f"""
You are the Localized Content Agent for Operating Systems.
Only answer if the topic '{topic}' is within the following syllabus:
{OS_SYLLABUS_CONTEXT}

If the topic is not in the syllabus, say "The topic '{topic}' is out of syllabus."

Otherwise, write a simplified explanation or story in Tamil that describes the concept of '{topic}' in an engaging, contextual way.
"""
    return call_gpt(prompt)

# === GPT Caller ===
def call_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# === WhatsApp Message Sender ===
def send_whatsapp_message(to, message):
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "text": {"body": message}
    }
    requests.post(url, headers=headers, json=payload)

# === Run ===
if __name__ == "__main__":
    app.run(port=5000)
