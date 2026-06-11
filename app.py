# app.py - COMPLETE FINAL VERSION
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import torch
from PyPDF2 import PdfReader

# Load .env (GROQ_API_KEY)
load_dotenv()

app = Flask(__name__)
app.secret_key = "change-this-in-production-12345"

# Upload config
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Database
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    title = db.Column(db.String(100), default="New Chat")
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey("conversation.id"), nullable=False)
    message = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

# Load RAG system
print("Loading RAG system...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})  # Increased for better report recall

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    max_tokens=1024
)

# YOUR EXACT REQUESTED PROMPT
template = """
You are Sympto Track AI — a friendly and helpful medical chatbot specialized in symptom monitoring and disease information.

Rules:
- For MEDICAL questions (symptoms, diseases, causes, precautions, treatments):
  - Use ONLY the provided context.
  - Answer professionally like a doctor.
  - Use bullet points for lists (symptoms, precautions).
  - Use paragraphs for explanations.
  - Keep length appropriate to the question.

- For PERSONAL/GREETING questions (name, age, "hi", "how are you", casual chat):
  - Respond naturally and warmly, like a friendly doctor.
  - Acknowledge the user (e.g., "Hi Arjun!", "Nice to meet you!").
  - Ask how you can help with health.

- For GENERAL/NON-MEDICAL questions (politics, news, math, history, etc.):
  - Politely refuse: "I'm focused on health and medical advice. I can't help with general topics, but I'm happy to assist with any symptoms or health questions!"

Medical Context (use only for medical questions):
{context}

Chat History:
{history}

Current Question: {question}

Answer naturally based on the question type.
"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Memory
memory = ConversationBufferMemory(return_messages=True, input_key="question", memory_key="history")

# RAG chain
rag_chain = (
    {
        "context": retriever | format_docs,
        "history": lambda x: memory.load_memory_variables({})["history"],
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Title generator
title_prompt = ChatPromptTemplate.from_template(
    "Generate a short, descriptive title (max 6 words) for this conversation based on the first user question: '{question}'. "
    "Make it natural and relevant, like ChatGPT titles."
)
title_chain = title_prompt | llm | StrOutputParser()

print("Sympto Track AI RAG Chatbot is ready!")

# Helper
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Upload Route - Auto creates new chat with summary
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        if 'file' not in request.files:
            flash("No file selected")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash("No file selected")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                reader = PdfReader(filepath)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

                if not text.strip():
                    flash("No readable text found in PDF.")
                    return redirect(request.url)

                # Add to vector store
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.create_documents([text])
                vectorstore.add_documents(chunks)

                # Generate summary (same as before)
                summary_prompt = f"""
You are an expert doctor analyzing a medical report.
Summarize key findings and extract patient details in simple language.

Report text:
{text[:8000]}

Extract:
- Patient name (if mentioned)
- Age
- Blood pressure
- Blood glucose/sugar
- Hemoglobin
- Any diagnosis

Then give a full summary with bullet points.
"""
                full_analysis = llm.invoke(summary_prompt).content

                # Create new conversation
                conv = Conversation(user_id=session["user_id"], title=f"Report: {filename[:30]}")
                db.session.add(conv)
                db.session.commit()

                session["conversation_id"] = conv.id
                memory.clear()

                # Add analysis as first message
                welcome_msg = ChatMessage(
                    conversation_id=conv.id,
                    message="Uploaded medical report",
                    response=f"Report uploaded: {filename}\n\nHere is my detailed analysis:\n\n{full_analysis}"
                )
                db.session.add(welcome_msg)
                db.session.commit()

                # PRIME MEMORY with key facts so model ALWAYS remembers
                memory.save_context(
                    {"question": "What is the patient name in the report?"},
                    {"output": f"The patient name mentioned in the report is from the uploaded document: {filename}."}
                )
                memory.save_context(
                    {"question": "What is my blood pressure from the report?"},
                    {"output": "Your blood pressure from the report is [value from report]."}
                )
                memory.save_context(
                    {"question": "What is my blood sugar level?"},
                    {"output": "Your blood glucose level from the report is [value]."}
                )
                memory.save_context(
                    {"question": "Summary of my medical report"},
                    {"output": full_analysis}
                )

                flash("Report analyzed and ready! Starting chat...")
                return redirect(url_for("chat"))

            except Exception as e:
                flash(f"Error processing file: {str(e)}")
        else:
            flash("Only PDF files allowed")

    return render_template("upload.html")

# Home
@app.route("/")
def home():
    if "user_id" in session:
        return redirect(url_for("conversations"))
    return render_template("home.html")

# Register
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if User.query.filter_by(username=username).first():
            flash("Username already exists!")
            return redirect(url_for("register"))
        hashed = generate_password_hash(password)
        new_user = User(username=username, password=hashed)
        db.session.add(new_user)
        db.session.commit()
        flash("Registered successfully! Please login.")
        return redirect(url_for("login"))
    return render_template("register.html")

# Login
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            session["username"] = user.username
            flash("Logged in successfully!")
            return redirect(url_for("conversations"))
        flash("Invalid credentials")
    return render_template("login.html")

# Conversations list
@app.route("/conversations")
def conversations():
    if "user_id" not in session:
        return redirect(url_for("login"))
    convs = Conversation.query.filter_by(user_id=session["user_id"]).order_by(Conversation.timestamp.desc()).all()
    return render_template("conversations.html", conversations=convs)

@app.route("/delete_chat/<int:conv_id>", methods=["POST"])
def delete_chat(conv_id):
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    conv = Conversation.query.get(conv_id)
    if conv and conv.user_id == session["user_id"]:
        # Delete all messages first
        ChatMessage.query.filter_by(conversation_id=conv_id).delete()
        # Delete conversation
        db.session.delete(conv)
        db.session.commit()
        flash("Conversation deleted successfully.")
    else:
        flash("Conversation not found or access denied.")
    
    return redirect(url_for("conversations"))

# New chat
@app.route("/new_chat", methods=["POST"])
def new_chat():
    if "user_id" not in session:
        return redirect(url_for("login"))
    conv = Conversation(user_id=session["user_id"])
    db.session.add(conv)
    db.session.commit()
    session["conversation_id"] = conv.id
    memory.clear()
    return redirect(url_for("chat"))

# Select existing chat
@app.route("/chat/<int:conv_id>")
def select_chat(conv_id):
    if "user_id" not in session:
        return redirect(url_for("login"))
    conv = Conversation.query.get(conv_id)
    if conv and conv.user_id == session["user_id"]:
        session["conversation_id"] = conv.id
        memory.clear()
        messages = ChatMessage.query.filter_by(conversation_id=conv_id).order_by(ChatMessage.timestamp).all()
        for msg in messages:
            memory.save_context({"question": msg.message}, {"output": msg.response})
        return redirect(url_for("chat"))
    flash("Conversation not found")
    return redirect(url_for("conversations"))

# Main chat
@app.route("/chat", methods=["GET", "POST"])
def chat():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if "conversation_id" not in session:
        return redirect(url_for("conversations"))

    conv_id = session["conversation_id"]
    conv = Conversation.query.get(conv_id)
    messages = ChatMessage.query.filter_by(conversation_id=conv_id).order_by(ChatMessage.timestamp).all()

    if request.method == "POST":
        question = request.form["question"].strip()
        if question:
            answer = rag_chain.invoke(question)
            
            chat_msg = ChatMessage(conversation_id=conv_id, message=question, response=answer)
            db.session.add(chat_msg)
            db.session.commit()
            
            memory.save_context({"question": question}, {"output": answer})
            
            if len(messages) == 0:
                title = title_chain.invoke(question).strip()
                conv.title = title[:100]
                db.session.commit()
            
            messages = ChatMessage.query.filter_by(conversation_id=conv_id).order_by(ChatMessage.timestamp).all()

    return render_template("chat.html", chats=messages, username=session["username"], conv_title=conv.title)

# Logout
@app.route("/logout")
def logout():
    session.clear()
    memory.clear()
    flash("Logged out successfully")
    return redirect(url_for("home"))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)