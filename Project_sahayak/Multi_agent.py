from flask import Flask, request
import requests
from PIL import Image
from io import BytesIO
import base64
from collections import defaultdict
from openai import OpenAI
from gtts import gTTS
from pydub import AudioSegment
import os

app = Flask(__name__)

# Configuration
VERIFY_TOKEN = "verifyme123"
ACCESS_TOKEN = "EAATCdCwA7KIBPB6WfSvnd7MQ37Xx83NqElYypoWoMrC7h9JqKTqdN2sD7B9116eKqvQORhY9Sn1dUkvugSw6i29bFN6433pmrm1tAL3PLRVs2CcR0v859gyvtWo6wTdOkMbUrWAdDSvXF8ZCtZB7s2MqPip668an8JBWGdlDOfZCWJrTh3cDGsKGSTitXbquMsQSXwkLSgpZAoZA6Y0nYxQP4lOXCWsp7TlScn9HEZB1heprUZD"
PHONE_NUMBER_ID = "699791233214714"
OPENAI_API_KEY = "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"

client = OpenAI(api_key=OPENAI_API_KEY)
user_memory = defaultdict(list)
MAX_HISTORY = 50  

# === Webhook verification ===
@app.route("/webhook", methods=["GET"])
def verify():
    if request.args.get("hub.verify_token") == VERIFY_TOKEN:
        return request.args.get("hub.challenge")
    return "Verification token mismatch", 403

# === Webhook handler ===
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

                    if msg.get("type") == "image":
                        media_id = msg["image"]["id"]
                        image_url = get_media_url(media_id)
                        caption = msg["image"].get("caption", "Describe the image.")
                        append_to_memory(sender, "user", caption)
                        reply = vision_reply(sender, image_url)
                        append_to_memory(sender, "assistant", reply)
                        send_whatsapp_message(sender, reply)

                    elif msg.get("type") == "text":
                        text = msg["text"]["body"]
                        append_to_memory(sender, "user", text)
                        reply = chat_reply(sender)
                        append_to_memory(sender, "assistant", reply)
                        send_whatsapp_message(sender, reply)

                    elif msg.get("type") == "audio":
                        media_id = msg["audio"]["id"]
                        audio_url = get_media_url(media_id)
                        reply = process_audio(sender, audio_url)
                        if reply:
                            append_to_memory(sender, "assistant", reply)
                            send_whatsapp_message(sender, reply)

                    else:
                        send_whatsapp_message(sender, "❗ Sorry, I can only understand text, images, or audio.")
                        
    except Exception as e:
        print("Error:", e)
    return "ok", 200

# === Append to memory ===
def append_to_memory(user_id, role, content):
    memory = user_memory[user_id]
    memory.append({"role": role, "content": content})
    if len(memory) > MAX_HISTORY:
        memory.pop(0)

# === Text Completion ===
def chat_reply(user_id):
    messages = [{"role": "system", "content": "You are a helpful assistant that answers WhatsApp user queries."}]
    messages += user_memory[user_id]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response.choices[0].message.content.strip()

# === Vision (Image + Caption) Completion ===
def vision_reply(user_id, image_url):
    latest_user_question = user_memory[user_id][-1]["content"]

    image_response = requests.get(image_url, headers={"Authorization": f"Bearer {ACCESS_TOKEN}"})
    image_data = image_response.content

    try:
        img = Image.open(BytesIO(image_data)).convert("RGB")
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        converted_image_data = buffer.read()
    except Exception as e:
        return f"❗ Failed to convert image: {str(e)}"

    base64_image = base64.b64encode(converted_image_data).decode("utf-8")
    data_url = f"data:image/png;base64,{base64_image}"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an assistant that can understand and describe images."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": latest_user_question},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }
        ],
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

# === Process Audio Message ===
def process_audio(user_id, audio_url):
    audio_data = requests.get(audio_url, headers={"Authorization": f"Bearer {ACCESS_TOKEN}"}).content
    input_audio_path = f"{user_id}.ogg"
    mp3_path = f"{user_id}.mp3"
    reply_path = f"{user_id}_reply.mp3"

    with open(input_audio_path, "wb") as f:
        f.write(audio_data)

    AudioSegment.from_file(input_audio_path).export(mp3_path, format="mp3")

    with open(mp3_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=f,
            model="whisper-1"
        ).text

    append_to_memory(user_id, "user", transcription)
    reply = chat_reply(user_id)

    tts = gTTS(reply)
    tts.save(reply_path)

    media_id = upload_audio_to_whatsapp(reply_path)
    send_whatsapp_audio(user_id, media_id)

    os.remove(input_audio_path)
    os.remove(mp3_path)
    os.remove(reply_path)
    return reply

# === Upload Audio File ===
def upload_audio_to_whatsapp(file_path):
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/media"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    files = {
        "file": (os.path.basename(file_path), open(file_path, "rb"), "audio/mpeg")
    }
    data = {
        "messaging_product": "whatsapp",
        "type": "audio"
    }
    res = requests.post(url, headers=headers, files=files, data=data)
    return res.json().get("id")

# === Send Audio Reply ===
def send_whatsapp_audio(to, media_id):
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "audio",
        "audio": {"id": media_id}
    }
    requests.post(url, headers=headers, json=payload)

# === Get media URL ===
def get_media_url(media_id):
    url = f"https://graph.facebook.com/v19.0/{media_id}"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    res = requests.get(url, headers=headers)
    return res.json().get("url")

# === Send Text Message ===
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
    response = requests.post(url, headers=headers, json=payload)
    print("Text Response:", response.json())

# === Run Server ===
if __name__ == "__main__":
    app.run(port=5000)
