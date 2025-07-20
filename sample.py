from flask import Flask, request
import requests
import os
import traceback
from collections import defaultdict
import re
from datetime import datetime, timedelta

# --- AI/ML Libraries ---
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# LangChain components for RAG
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- PDF Generation Library ---
from fpdf import FPDF 
from io import BytesIO # Crucial for handling binary PDF data in memory

# Suppress potential OpenMP runtime warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)

# --- Configuration ---
VERIFY_TOKEN = "verifyme123"

# Your WhatsApp Business API Access Token.
# This MUST be a long-lived System User Access Token for production.
# Replace with your actual token.
ACCESS_TOKEN = "EAAR4pFKW5JEBPPlYiR9QBQd13jP8BC1Neh1U2bXAZAr6qSZCVojV0U8I1Hv8TScWeCVvgmHkv5dJ92yhR3oHLj8yU5xLy94LYpDjKe84sObkZClRl577ENstSovYPyBd7ckOV0kMwkeNu2Vwc1rPZA1dPimizDDm1ZCqk4sxCCszXUc5BJnZBR59Hd5NwcMieFQrzsn89m91gC5YW09cwI6LzSfTsWDoGoWFFpeKlSKwkZD" 

# Your WhatsApp Business Account's Phone Number ID.
# Replace with your actual Phone Number ID.
PHONE_NUMBER_ID = "717383254790727"

# Configure Google Gemini API. (Used for content generation and embeddings)
genai.configure(api_key="AIzaSyCnU2gMleVnAScCuaQPhz67n63imPPaWS8") # !!! IMPORTANT: REPLACE WITH YOUR ACTUAL GEMINI API KEY !!!


user_memory = defaultdict(list)
MAX_HISTORY = 50

# --- Semantic Similarity Examples for Routing ---
# OS_EXAMPLES remain the same for Operating Systems content retrieval
OS_EXAMPLES = [
    "What is an operating system?", "Explain process scheduling.", "Tell me about deadlocks in OS.",
    "How does virtual memory work?", "What are the types of file systems?", "Describe the role of the kernel.",
    "What is multitasking?", "How does memory management happen in an OS?", "What is CPU scheduling?",
    "Explain paging.", "What is a semaphore in OS?", "Discuss thread management.",
    "What are system calls?", "Explain storage management.", "What is a logical address space?",
    "What is a process control block?", "Describe context switching.", "Explain segmentation.",
    "What is I/O management?", "How does buffering work in OS?", "What is thrashing?",
    "Explain mutual exclusion.", "What are producers-consumers problem?", "Describe inter-process communication.",
    "What is fragmentation?",
    "What is the OS syllabus?", "Can you provide details on the operating system syllabus?",
    "Describe my OS syllabus.", "Give me an overview of the OS course content.",
    "What topics are covered in the OS subject?", "Show me the syllabus for Operating Systems.",
    "List the units in the OS syllabus?", "What are the main topics in the OS syllabus?",
    "Can you summarize the OS syllabus for me?", "What is the structure of the OS syllabus?",
    "What are the key areas in the OS syllabus?", "How is the OS syllabus organized?",
    "Tell me about the course outline for OS."
]

# WORKSHEET_GENERATION_EXAMPLES for PDF creation intent
WORKSHEET_GENERATION_EXAMPLES = [
    "Generate a worksheet on deadlocks.",
    "Create a PDF worksheet for memory management.",
    "Give me a short answer worksheet on CPU scheduling.",
    "Make a worksheet with 5 fill-in-the-blanks on virtual memory.",
    "Can you generate a practice sheet for OS concepts?",
    "Generate a PDF for operating systems review.",
    # --- NEW ADDITIONS FOR SPECIFIC WORKSHEET TYPES ---
    "Generate 10 multiple choice questions worksheet on processes.",
    "Create an MCQ worksheet for file systems.",
    "Give me 5 short answer questions on concurrency.",
    "Generate a short answer worksheet about synchronization.",
    "Make 3 long answer questions on memory allocation strategies.",
    "Give me an essay question worksheet on virtual memory benefits.",
    "Generate 2 numerical questions on CPU scheduling algorithms.",
    "Create a worksheet with calculation problems about page replacement."
]

embeddings_model_for_classification = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
os_example_embeddings = None
worksheet_generation_example_embeddings = None

SIMILARITY_THRESHOLD_OS = 0.65 
SIMILARITY_THRESHOLD_WORKSHEET = 0.65 

# === Utility Functions (Defined before they are called by webhook or other functions) ===

def append_to_memory(user_id, role, content):
    """
    Appends a message (from user or assistant) to a user's conversation history.
    It maintains a maximum history length by removing the oldest message if exceeded.
    """
    user_memory[user_id].append({"role": role, "content": str(content)})
    if len(user_memory[user_id]) > MAX_HISTORY:
        user_memory[user_id].pop(0)

def send_whatsapp_message(to, message):
    """
    Sends a TEXT message to a WhatsApp number via the Meta WhatsApp Cloud API.
    """
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}", "Content-Type": "application/json"}
    payload = {"messaging_product": "whatsapp", "to": to, "text": {"body": message}}
    try:
        res = requests.post(url, headers=headers, json=payload)
        res.raise_for_status()
        print("WhatsApp text message sent successfully:", res.json())
    except Exception as e:
        print("Failed to send WhatsApp text message:", res.text if hasattr(res, 'text') else "No response text available")
        traceback.print_exc()

def send_whatsapp_document(to, file_bytes, filename):
    """
    Uploads a file to WhatsApp's media server and sends it as a document message.
    """
    media_url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/media"
    # Headers for authorization only. requests will set Content-Type for multipart/form-data.
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
    }

    # This structure correctly prepares the multipart/form-data for WhatsApp's media upload endpoint.
    # 'file' contains the binary data.
    # 'messaging_product' and 'type' are REQUIRED form fields for WhatsApp media uploads.
    files_and_data = {
        'file': (filename, file_bytes, 'application/pdf'), # (filename, file_content_bytes, mime_type)
        'messaging_product': (None, 'whatsapp'),             # THIS IS THE CRITICAL FIX for the error!
        'type': (None, 'document')                           # Also required for media upload type
    }

    # --- Step 1: Upload the PDF file to WhatsApp's media server ---
    try:
        # requests will automatically set the 'Content-Type: multipart/form-data' header.
        media_res = requests.post(
            media_url, 
            headers=headers, 
            files=files_and_data # <<< THIS IS THE CORRECT WAY TO SEND MULTIPART/FORM-DATA
        )
        media_res.raise_for_status() # Raise error for bad status codes
        media_id = media_res.json().get("id")

        if not media_id:
            raise Exception("Failed to get media ID from WhatsApp upload response.")
        print(f"Successfully uploaded PDF to WhatsApp media server. Media ID: {media_id}")
    except Exception as e:
        print(f"Failed to upload PDF to WhatsApp media server: {e}")
        print("Media upload response:", media_res.text if 'media_res' in locals() else "No response available (request failed before response)")
        traceback.print_exc()
        return {"result": "Sorry, I couldn't upload the PDF to WhatsApp's server.", "source": "whatsapp_media_error"}

    # --- Step 2: Send the document message using the media ID ---
    message_url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    message_headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json" # This remains JSON for the message payload
    }
    message_payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "document",
        "document": {
            "id": media_id,
            "filename": filename # The name of the file as it appears in WhatsApp
        }
    }

    try:
        msg_res = requests.post(message_url, headers=message_headers, json=message_payload)
        msg_res.raise_for_status()
        print("WhatsApp document message sent successfully:", msg_res.json())
        return {"result": f"I've generated a worksheet PDF and sent it to you: '{filename}'", "source": "whatsapp_document_sent"}
    except Exception as e:
        print(f"Failed to send WhatsApp document message: {e}")
        print("Message send response:", msg_res.text if 'msg_res' in locals() else "No response available (request failed before response)")
        traceback.print_exc()
        return {"result": "Sorry, I generated the PDF but couldn't send it via WhatsApp.", "source": "whatsapp_send_error"}

# === Vector Index (build once) ===
def build_os_vector_index():
    print("Loading documents for OS vector index...")
    loaders = [
        UnstructuredFileLoader("data/OS_Syllabus.pdf"),
        UnstructuredFileLoader("data/OS_Unit_I_Overview.pptx"),
        UnstructuredFileLoader("data/OS_Unit_II_Process_Management.pptx"),
        UnstructuredFileLoader("data/OS_Unit_II_Process_Scheduling.pptx"),
        UnstructuredFileLoader("data/OS_Unit_III_Deadlocks.pptx"),
        UnstructuredFileLoader("data/OS_Unit_IV_Memory_management.pptx"),
        UnstructuredFileLoader("data/OS_Unit_IV_Virtual_Memory.pptx"),
        UnstructuredFileLoader("data/OS_Unit_V_File_System_Implementation.pptx"),
        UnstructuredFileLoader("data/OS_Unit_V_Storage_management.pptx"),
    ]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = []
    for loader in loaders:
        try:
            raw_docs = loader.load()
            split_docs = text_splitter.split_documents(raw_docs)
            documents.extend(split_docs)
            print(f"Loaded and split {len(raw_docs)} documents from {loader.file_path}")
        except Exception as e:
            print(f"Error loading {loader.file_path}: {e}")
            traceback.print_exc()

    if not documents:
        print("No documents were loaded successfully for vector index. FAISS will be empty.")
        return

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(documents, embeddings)
    db.save_local("vector_index/os_docs")
    print("OS Vector Index built successfully.")

def query_os_subject(question):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = FAISS.load_local("vector_index/os_docs", embeddings, allow_dangerous_deserialization=True)
        
        retriever = db.as_retriever(search_kwargs={"k": 1}) 
        docs = retriever.invoke(question) 
        
        print(f"\n--- Inside query_os_subject for: '{question}' ---")
        if docs:
            print(f"Retrieved {len(docs)} documents.")
            for i, doc in enumerate(docs):
                print(f"  Doc {i+1} content snippet: '{doc.page_content[:150].strip()}...'")
                if not doc.page_content.strip():
                    print(f"  WARNING: Doc {i+1} has empty content.")
        else:
            print("No documents retrieved from FAISS.")

        if not docs or all(len(doc.page_content.strip()) == 0 for doc in docs):
            print(f"No relevant OS documents found or documents are empty for '{question}'. Signaling fallback.")
            return {"result": "No relevant content found in OS database.", "source": "fallback_needed"}

        context = "\n\n".join([doc.page_content for doc in docs])
        
        sources = []
        for doc in docs:
            if 'source' in doc.metadata:
                filename = os.path.basename(doc.metadata['source'])
                sources.append(filename)
        
        source_info = ""
        if sources:
            unique_sources = list(set(sources))
            source_info = "\n\n(Source: " + ", ".join(unique_sources) + ")"

        prompt = f"Based on the following context, answer the question accurately and concisely. If the context does not contain enough information to answer the question fully, state that directly:\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        final_answer = response.text + source_info

        print(f"Gemini response (RAG): '{response.text}'")
        print(f"Final answer with source: '{final_answer}'")
        print(f"--- Exiting query_os_subject for: '{question}' ---")

        return {"result": final_answer, "source": "db"}

    except Exception as e:
        print("Error during OS subject query or vector store loading:", e)
        traceback.print_exc()
        print(f"Error in OS retrieval for '{question}'. Signaling fallback.")
        return {"result": "Error during database lookup.", "source": "fallback_needed"}

def query_gemini(question, history):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        chat = model.start_chat(history=history)
        response = chat.send_message(question)
        return {"result": response.text}
    except Exception as e:
        print("Gemini general query error:", e)
        traceback.print_exc()
        return {"result": "Sorry, I encountered an issue while processing your general question with Gemini. Please try again later."}

def generate_mcq_questions_text(topic, num_questions, history):
    """
    Generates MCQ questions in plain text format using Gemini.
    Leverages RAG if the topic is OS-related by first retrieving context.
    """
    print(f"Attempting to generate {num_questions} TEXT MCQs on topic: '{topic}'")
    
    is_os_topic = False
    if os_example_embeddings is not None:
        topic_embedding = embeddings_model_for_classification.embed_query(topic)
        topic_embedding_reshaped = np.array(topic_embedding).reshape(1, -1)
        
        similarities = cosine_similarity(topic_embedding_reshaped, os_example_embeddings)
        max_similarity = np.max(similarities)
        
        if max_similarity > SIMILARITY_THRESHOLD_OS:
            is_os_topic = True
            print(f"Topic '{topic}' classified as OS-related for MCQ generation (Similarity: {max_similarity:.4f}). Will attempt RAG.")
        else:
            print(f"Topic '{topic}' classified as general for MCQ generation (Similarity: {max_similarity:.4f}). Will use general Gemini.")

    context = ""
    source_info = ""
    if is_os_topic:
        print(f"Retrieving context for MCQ topic '{topic}' from OS docs...")
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            db = FAISS.load_local("vector_index/os_docs", embeddings, allow_dangerous_deserialization=True)
            retriever = db.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(topic)

            if docs and not all(len(doc.page_content.strip()) == 0 for doc in docs):
                context = "\n\n".join([doc.page_content for doc in docs])
                unique_sources = list(set([os.path.basename(d.metadata['source']) for d in docs if 'source' in d.metadata]))
                if unique_sources:
                    source_info = f"\n\n(Context from: {', '.join(unique_sources)})"
                print(f"Successfully retrieved context for '{topic}'.")
            else:
                print(f"No relevant OS context found for '{topic}'. Using general Gemini for MCQ generation.")
        except Exception as e:
            print(f"Error retrieving context for MCQ topic '{topic}': {e}. Using general Gemini.")
            context = ""

    mcq_prompt = f"Generate {num_questions} ONLY multiple-choice questions on the topic of '{topic}'. " \
                 "For each question, provide 4 options (A, B, C, D) and clearly indicate the correct answer. " \
                 "Ensure the questions are suitable for a quiz. " \
                 "Format each question as: 'Question Number. Question Text\\n A) Option A\\n B) Option B\\n C) Option C\\n D) Option D\\n Correct Answer: X'. " \
                 "Do NOT include any other question types like fill-in-the-blanks or short answer questions. Provide ONLY the questions and answers."
    
    if context:
        mcq_prompt = f"Based on the following context:\n\n{context}\n\n" + mcq_prompt
        
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        chat = model.start_chat(history=history)
        response = chat.send_message(mcq_prompt)
        
        generated_questions = response.text + source_info

        print(f"Generated textual MCQs for '{topic}':\n{generated_questions[:500]}...")

        return {"result": generated_questions, "source": "generated_form_questions_text"}

    except Exception as e:
        print(f"Error generating TEXT MCQs for '{topic}': {e}")
        traceback.print_exc()
        return {"result": "Sorry, I couldn't generate the quiz questions in text format right now. Please try again later.", "source": "error"}


# --- Worksheet Generation and PDF Creation Functions ---
def generate_worksheet_content_text(topic, num_items, worksheet_type, history):
    """
    Generates worksheet content (e.g., short answers, fill-in-the-blanks) using Gemini.
    Leverages RAG if the topic is OS-related.
    """
    print(f"Attempting to generate {num_items} {worksheet_type} items on topic: '{topic}'")
    
    is_os_topic = False
    if os_example_embeddings is not None:
        topic_embedding = embeddings_model_for_classification.embed_query(topic)
        topic_embedding_reshaped = np.array(topic_embedding).reshape(1, -1)
        
        similarities = cosine_similarity(topic_embedding_reshaped, os_example_embeddings)
        max_similarity = np.max(similarities)
        
        if max_similarity > SIMILARITY_THRESHOLD_OS:
            is_os_topic = True
            print(f"Topic '{topic}' classified as OS-related for worksheet generation (Similarity: {max_similarity:.4f}). Will attempt RAG.")
        else:
            print(f"Topic '{topic}' classified as general for MCQ generation (Similarity: {max_similarity:.4f}). Will use general Gemini.")

    context = ""
    source_info = ""
    if is_os_topic:
        print(f"Retrieving context for worksheet topic '{topic}' from OS docs...")
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            db = FAISS.load_local("vector_index/os_docs", embeddings, allow_dangerous_deserialization=True)
            retriever = db.as_retriever(search_kwargs={"k": 4})
            docs = retriever.invoke(topic)

            if docs and not all(len(doc.page_content.strip()) == 0 for doc in docs):
                context = "\n\n".join([doc.page_content for doc in docs])
                unique_sources = list(set([os.path.basename(d.metadata['source']) for d in docs if 'source' in d.metadata]))
                if unique_sources:
                    source_info = f"\n\n(Context from: {', '.join(unique_sources)})"
                print(f"Successfully retrieved context for '{topic}'.")
            else:
                print(f"No relevant OS context found for '{topic}'. Using general Gemini for MCQ generation.")
        except Exception as e:
            print(f"Error retrieving context for MCQ topic '{topic}': {e}. Using general Gemini.")
            context = ""

    # Craft the prompt for Gemini to generate worksheet content based on worksheet_type
    worksheet_prompt_base = f"Generate {num_items} questions/items on the topic of '{topic}'. " \
                           f"Format them clearly with numbers. Ensure they are suitable for a student worksheet. " \
                           f"Provide ONLY this type of question. Do NOT include other types like multiple-choice, fill-in-the-blanks (unless requested), or short answer (unless requested)."
    
    worksheet_prompt = "" # Initialize here

    # Handle different worksheet types
    if worksheet_type == "fill-in-the-blanks":
        worksheet_prompt = f"{worksheet_prompt_base} Use underscores (______) for blanks. Do not provide answers."
    elif worksheet_type == "short answer":
        worksheet_prompt = f"{worksheet_prompt_base} Pose direct short answer questions. Do not provide answers."
    elif worksheet_type == "essay prompts" or worksheet_type == "long answer":
        worksheet_prompt = f"{worksheet_prompt_base} Pose detailed essay prompts. Do not provide answers."
    elif worksheet_type == "multiple choice" or worksheet_type == "mcq":
        # Reuse the strict MCQ generation logic here
        mcq_result = generate_mcq_questions_text(topic, num_items, history)
        if mcq_result["source"] == "generated_form_questions_text": # This source name is a bit misleading here, but it works
            return {"result": mcq_result["result"], "source": "generated_worksheet_text"}
        else:
            return mcq_result # Propagate error from MCQ generation
    elif worksheet_type == "numerical" or worksheet_type == "calculations":
         worksheet_prompt = f"{worksheet_prompt_base} Generate numerical problems requiring calculations. For each problem, briefly describe the scenario and the calculation required. Do not provide answers."
    else: # Default if type isn't clearly specified or recognized
        worksheet_prompt = f"{num_items} general questions on the topic of '{topic}'. {worksheet_prompt_base}"
    
    if context and worksheet_type != "multiple choice" and worksheet_type != "mcq": # Avoid double-context if reusing mcq function
        worksheet_prompt = f"Based on the following context:\n\n{context}\n\n" + worksheet_prompt
        
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        chat = model.start_chat(history=history)
        response = chat.send_message(worksheet_prompt)
        
        generated_content = response.text + source_info

        print(f"Generated worksheet content for '{topic}':\n{generated_content[:500]}...")

        return {"result": generated_content, "source": "generated_worksheet_text"}

    except Exception as e:
        print(f"Error generating TEXT MCQs for '{topic}': {e}")
        traceback.print_exc()
        return {"result": "Sorry, I couldn't generate the quiz questions in text format right now. Please try again later.", "source": "error"}

def create_pdf_locally(title, content):
    """
    Creates a PDF from the given content and returns its bytes.
    This function does NOT upload to Google Drive or any external storage.
    """
    print(f"Attempting to create PDF locally for '{title}'.")

    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.set_font("Helvetica", 'B', 16)
    pdf.multi_cell(0, 10, title, align='C')
    pdf.ln(10)

    pdf.set_font("Helvetica", size = 11)
    
    usable_width = pdf.w - pdf.l_margin - pdf.r_margin 

    lines = content.split('\n')
    for line in lines:
        stripped_line = line.strip()
        if stripped_line:
            try:
                pdf.multi_cell(usable_width - 0.1, 7, stripped_line)
                
                if re.match(r'^\s*\d+\.\s*', stripped_line) or stripped_line.startswith(('A.', 'B.', 'C.', 'D.')):
                    pdf.ln(2) 
                else:
                    pdf.ln(1)
            except Exception as e:
                print(f"Warning: FPDF rendering error for line: '{stripped_line[:100]}...'. Error: {e}")
                pdf.add_page()
                pdf.set_font("Helvetica", size = 11)
                pdf.multi_cell(0, 7, "[Error rendering content, some text might be missing or malformed in PDF.]")
                pdf.ln(5)
        else:
            pdf.ln(3)

    pdf_output_bytes = pdf.output() 
    
    print(f"Successfully created PDF content in memory for '{title}'.")
    return pdf_output_bytes


# --- Function to send PDF as a WhatsApp Document Message ---
def send_whatsapp_document(to, file_bytes, filename):
    """
    Uploads a file to WhatsApp's media server and sends it as a document message.
    """
    media_url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/media"
    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
    }
    
    # This structure correctly prepares the multipart/form-data for WhatsApp's media upload endpoint.
    # 'file' contains the binary data.
    # 'messaging_product' and 'type' are REQUIRED form fields for WhatsApp media uploads.
    files_and_data = {
        'file': (filename, file_bytes, 'application/pdf'), 
        'messaging_product': (None, 'whatsapp'),             # THIS IS THE CRITICAL FIX for the error!
        'type': (None, 'document')                           # Also required for media upload type
    }
    
    # --- Step 1: Upload the PDF file to WhatsApp's media server ---
    try:
        # requests will automatically set the 'Content-Type: multipart/form-data' header.
        media_res = requests.post(
            media_url, 
            headers=headers, 
            files=files_and_data # <<< THIS IS THE CORRECT WAY TO SEND MULTIPART/FORM-DATA
        )
        media_res.raise_for_status() 
        media_id = media_res.json().get("id")
        
        if not media_id:
            raise Exception("Failed to get media ID from WhatsApp upload response.")
        print(f"Successfully uploaded PDF to WhatsApp media server. Media ID: {media_id}")
    except Exception as e:
        print(f"Failed to upload PDF to WhatsApp media server: {e}")
        print("Media upload response:", media_res.text if 'media_res' in locals() else "No response available (request failed before response)")
        traceback.print_exc()
        return {"result": "Sorry, I couldn't upload the PDF to WhatsApp's server.", "source": "whatsapp_media_error"}

    # --- Step 2: Send the document message using the media ID ---
    message_url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    message_headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json" 
    }
    message_payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "document",
        "document": {
            "id": media_id,
            "filename": filename 
        }
    }
    
    try:
        msg_res = requests.post(message_url, headers=message_headers, json=message_payload)
        msg_res.raise_for_status()
        print("WhatsApp document message sent successfully:", msg_res.json())
        return {"result": f"I've generated a worksheet PDF and sent it to you: '{filename}'", "source": "whatsapp_document_sent"}
    except Exception as e:
        print(f"Failed to send WhatsApp document message: {e}")
        print("Message send response:", msg_res.text if 'msg_res' in locals() else "No response available (request failed before response)")
        traceback.print_exc()
        return {"result": "Sorry, I generated the PDF but couldn't send it via WhatsApp.", "source": "whatsapp_send_error"}

# === Webhook & WhatsApp Routing ===
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
                        append_to_memory(sender, "user", text) 

                        user_intent = "general"
                        
                        gemini_history = []
                        for item in user_memory[sender]:
                            gemini_history.append({"role": "user" if item["role"] == "user" else "model", "parts": [item["content"]]})
                        if gemini_history and gemini_history[-1]["role"] == "user" and gemini_history[-1]["parts"][0] == text:
                            gemini_history.pop()

                        # Determine intent using semantic similarity for defined features (OS & Worksheet)
                        # Removed Forms/Classroom from this check as those features are out.
                        if os_example_embeddings is not None and worksheet_generation_example_embeddings is not None:
                            
                            user_embedding = embeddings_model_for_classification.embed_query(text)
                            user_embedding_reshaped = np.array(user_embedding).reshape(1, -1)
                            
                            max_similarity_os = np.max(cosine_similarity(user_embedding_reshaped, os_example_embeddings))
                            max_similarity_worksheet = np.max(cosine_similarity(user_embedding_reshaped, worksheet_generation_example_embeddings))
                            
                            print(f"User query '{text}' -- OS Sim: {max_similarity_os:.4f} (Thresh {SIMILARITY_THRESHOLD_OS}) | Worksheet Sim: {max_similarity_worksheet:.4f} (Thresh {SIMILARITY_THRESHOLD_WORKSHEET})")

                            # --- Intent Classification Logic (Prioritized) ---
                            # 1. Prioritize Worksheet Generation
                            if max_similarity_worksheet > SIMILARITY_THRESHOLD_WORKSHEET and \
                               (max_similarity_worksheet >= max_similarity_os): # Worksheet dominates OS if both strong
                                user_intent = "worksheet_generation"
                                print(f"Intent Classified: Worksheet Generation (Strongest & above threshold)")
                            # 2. Then, prioritize OS query
                            elif max_similarity_os > SIMILARITY_THRESHOLD_OS: 
                                user_intent = "os_query"
                                print(f"Intent Classified: OS Query (OS strong enough)")
                            # 3. Default to general
                            else:
                                user_intent = "general"
                                print(f"Intent Classified: General (No specific strong intent)")
                            # --- End Intent Classification Logic ---

                        else:
                            print("Warning: Example embeddings not initialized. Defaulting to general query.")
                            user_intent = "general"

                        reply_message = "" # Initialize variable to hold the message that might be sent to WhatsApp

                        if user_intent == "os_query":
                            os_result = query_os_subject(text)
                            print(f"Result from query_os_subject: {os_result}")
                            reply_message = os_result["result"]
                            send_whatsapp_message(sender, reply_message) # Send as text message (OS RAG output)
                        
                        elif user_intent == "worksheet_generation":
                            # Extract details for worksheet creation
                            # --- MODIFIED REGEX FOR WORKSHEET TYPE EXTRACTION ---
                            num_items_match = re.search(r'(\d+)\s+(?:fill-in-the-blanks|short answer|long answer|essay prompts|numerical|calculation|mcq|multiple choice|questions?|items?)', text, re.IGNORECASE)
                            worksheet_type_match = re.search(r'(fill-in-the-blanks|short answer|long answer|essay prompts|numerical|calculation|mcq|multiple choice|questions|items)', text, re.IGNORECASE)
                            # --- END MODIFIED REGEX ---
                            worksheet_topic_match = re.search(r'(?:on|in|about)\s+(.+)', text, re.IGNORECASE)

                            num_items = 5 # Default items
                            worksheet_type = "questions" # Default type
                            worksheet_topic = "Operating Systems Concepts" # Default topic

                            if num_items_match:
                                try:
                                    num_items = int(num_items_match.group(1))
                                    if num_items < 1: num_items = 1
                                    if num_items > 20: num_items = 20
                                except ValueError:
                                    pass
                            if worksheet_type_match:
                                worksheet_type = worksheet_type_match.group(1).lower()
                                # Normalize worksheet_type for consistent handling
                                if worksheet_type == "multiple choice": worksheet_type = "mcq"
                                if worksheet_type == "calculation": worksheet_type = "numerical"
                                if worksheet_type == "items": worksheet_type = "questions" # Map 'items' to 'questions'
                            if worksheet_topic_match:
                                worksheet_topic = worksheet_topic_match.group(1).strip()
                            else:
                                generic_topic_match = re.search(r'(?:worksheet|practice sheet|pdf) on\s+(.+)', text, re.IGNORECASE)
                                if generic_topic_match:
                                    worksheet_topic = generic_topic_match.group(1).strip()

                            print(f"Extracted for Worksheet: Items={num_items}, Type='{worksheet_type}', Topic='{worksheet_topic}'")

                            # Call the generation function. This handles RAG internally.
                            worksheet_content_result = generate_worksheet_content_text(worksheet_topic, num_items, worksheet_type, gemini_history)

                            if worksheet_content_result["source"] == "generated_worksheet_text":
                                generated_worksheet_text = worksheet_content_result["result"]
                                content_for_pdf = generated_worksheet_text.split('\n\n(Context from:')[0].strip()

                                pdf_title = f"OS Bot Worksheet: {worksheet_topic.title()} - {worksheet_type.title()}"
                                
                                pdf_bytes = create_pdf_locally(title=pdf_title, content=content_for_pdf)
                                
                                if pdf_bytes: # If PDF creation was successful
                                    send_doc_result = send_whatsapp_document(
                                        to=sender, 
                                        file_bytes=pdf_bytes, 
                                        filename=f"{pdf_title.replace(':', '-').replace('/', '-')}.pdf"
                                    )
                                    reply_message = send_doc_result["result"] 
                                else:
                                    reply_message = "Sorry, I generated the worksheet content but failed to create the PDF."
                            else:
                                reply_message = worksheet_content_result["result"] 

                            # The reply message will already be sent by send_whatsapp_document or send_whatsapp_message
                            # No need for an extra send_whatsapp_message call here.

                        else: # General intent
                            result_from_gemini = query_gemini(text, gemini_history)
                            reply_message = result_from_gemini["result"]
                            send_whatsapp_message(sender, reply_message) # Send as text message

                        # Append bot's final reply to memory (important for conversation history)
                        append_to_memory(sender, "assistant", reply_message)

    except Exception as e:
        print("Webhook processing error:", e)
        traceback.print_exc()

    return "ok", 200

@app.route("/webhook", methods=["GET"])
def verify():
    if request.args.get("hub.verify_token") == VERIFY_TOKEN:
        print("Webhook verified successfully!")
        return request.args.get("hub.challenge")
    print("Verification token mismatch.")
    return "Verification token mismatch", 403

@app.route("/status", methods=["GET"])
def status():
    return {"status": "running"}, 200

# === Application Entry Point ===
if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("vector_index", exist_ok=True)

    print("Embedding OS example questions for classification...")
    os_example_embeddings = embeddings_model_for_classification.embed_documents(OS_EXAMPLES)
    os_example_embeddings = np.array(os_example_embeddings)
    print(f"Embedded {len(OS_EXAMPLES)} OS examples for routing.")

    print("Embedding Worksheet Generation example questions for classification...")
    worksheet_generation_example_embeddings = embeddings_model_for_classification.embed_documents(WORKSHEET_GENERATION_EXAMPLES)
    worksheet_generation_example_embeddings = np.array(worksheet_generation_example_embeddings)
    print(f"Embedded {len(WORKSHEET_GENERATION_EXAMPLES)} Worksheet Generation examples for routing.")

    if not os.path.exists("vector_index/os_docs"):
        print("Building OS vector index for the first time...")
        build_os_vector_index()
    else:
        print("OS Vector Index already exists. Skipping build.")

    print("Starting Flask app...")
    app.run(port=5000, debug=False)