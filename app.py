import os
import tempfile
import json
from flask import Flask, render_template, request, jsonify, Response, session, abort
from flask_cors import CORS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
import requests
from sentence_transformers import SentenceTransformer
import warnings
import uuid
import fitz
import base64


app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.secret_key = os.urandom(24)  # Required for session management

# Initialize embeddings model with strict offline mode
local_model_path = os.path.join(os.path.dirname(__file__), "models")
if not os.path.exists(local_model_path):
    raise RuntimeError(f"Model not found in {local_model_path}. Please run download_model.py first.")


os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

print(f"Loading model from {local_model_path}")
model = SentenceTransformer(local_model_path)

embeddings = HuggingFaceEmbeddings(
    model_name=local_model_path,
    cache_folder=local_model_path,
    model_kwargs={'device': 'cpu'}
)

def get_pdf_text(pdf_file):
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp.pdf")
    pdf_file.save(temp_path)
    loader = PyPDFLoader(temp_path)
    pages = loader.load()
    all_page_chunks = [] 
    for page in pages: 
        page_text = page.page_content
        page_chunks = get_text_chunks(page_text)
        all_page_chunks.extend(page_chunks) 
    return all_page_chunks 

def get_docx_text(docx_file):
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp.docx")
    docx_file.save(temp_path)
    loader = Docx2txtLoader(temp_path)
    documents = loader.load()
    all_chunks = []
    for doc in documents:
        chunks = get_text_chunks(doc.page_content)
        all_chunks.extend(chunks)
    return all_chunks

def get_document_text_from_path(file_path):
    if file_path.lower().endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file type")
        
    documents = loader.load()
    all_chunks = []
    for doc in documents:
        chunks = get_text_chunks(doc.page_content)
        all_chunks.extend(chunks)
    return all_chunks

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks):
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# --- Preload all documents at startup ---
preloaded_vector_store = None
PRELOADED_PDF_FOLDER = "preloaded_pdfs"

# Initialize storage for individual SOP vector stores
sop_vector_stores = {}

@app.route('/get_sop_folders')
def get_sop_folders():
    """Get list of available folders in preloaded_pdfs directory"""
    sop_dir = os.path.join(app.root_path, 'preloaded_pdfs')
    folders = []
    
    if os.path.exists(sop_dir):
        # Get immediate subdirectories
        folders = [d for d in os.listdir(sop_dir) 
                  if os.path.isdir(os.path.join(sop_dir, d))]
    
    return jsonify(folders)

@app.route('/get_subfolders/<folder>')
def get_subfolders(folder):
    """Get immediate subfolders of a given folder inside preloaded_pdfs"""
    sop_dir = os.path.join(app.root_path, 'preloaded_pdfs', folder)
    subfolders = []
    if os.path.exists(sop_dir):
        subfolders = [d for d in os.listdir(sop_dir)
                      if os.path.isdir(os.path.join(sop_dir, d))]
    return jsonify(subfolders)

@app.route('/get_sops/<folder>', defaults={'subfolder': None})
@app.route('/get_sops/<folder>/<subfolder>')
def get_sops_in_folder(folder, subfolder):
    """Get list of SOPs (PDFs) directly in a specific folder or subfolder (not recursive)"""
    if subfolder:
        sop_dir = os.path.join(app.root_path, 'preloaded_pdfs', folder, subfolder)
        category = f"{folder}/{subfolder}"
    else:
        sop_dir = os.path.join(app.root_path, 'preloaded_pdfs', folder)
        category = folder
    sops = []
    if os.path.exists(sop_dir):
        for file in os.listdir(sop_dir):
            file_path = os.path.join(sop_dir, file)
            if os.path.isfile(file_path) and file.endswith('.pdf'):
                try:
                    doc = fitz.open(file_path)
                    text = doc[0].get_text()
                    description = ' '.join(text.split('\n')[:3])
                    doc.close()
                except:
                    description = "No description available"
                sops.append({
                    'id': base64.urlsafe_b64encode(file_path.encode()).decode(),
                    'title': file,
                    'category': category,
                    'description': description
                })
    else:
        abort(404)
    return jsonify(sops)

# Keep the original get_sops route for backward compatibility
@app.route('/get_sops')
def get_sops():
    """Get list of available SOPs with their categories"""
    # This route is maintained for backward compatibility
    sop_dir = os.path.join(app.root_path, 'preloaded_pdfs')
    sops = []
    
    if os.path.exists(sop_dir):
        for root, _, files in os.walk(sop_dir):
            for file in files:
                if file.endswith('.pdf'):
                    category = os.path.basename(root)
                    pdf_path = os.path.join(root, file)
                    try:
                        doc = fitz.open(pdf_path)
                        text = doc[0].get_text()
                        description = ' '.join(text.split('\n')[:3])
                        doc.close()
                    except:
                        description = "No description available"
                    
                    sops.append({
                        'id': base64.urlsafe_b64encode(pdf_path.encode()).decode(),
                        'title': file,
                        'category': category,
                        'description': description
                    })
                    
    return jsonify(sops)

@app.route('/select_sop/<sop_id>')
def select_sop(sop_id):
    """Get summary of selected SOP and prepare it for querying"""
    try:
        pdf_path = base64.urlsafe_b64decode(sop_id.encode()).decode()
        if not os.path.exists(pdf_path):
            return jsonify({'error': 'SOP not found'}), 404

        # Create vector store for this specific SOP if not already created
        if sop_id not in sop_vector_stores:
            try:
                chunks = get_document_text_from_path(pdf_path)
                sop_vector_stores[sop_id] = create_vector_store(chunks)
            except Exception as e:
                return jsonify({'error': f'Error processing SOP: {str(e)}'}), 500

        # Store the selected SOP info in session
        session['current_sop_id'] = sop_id
        session['current_sop_path'] = pdf_path
        
        return jsonify({
            'message': 'SOP processed successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Variable for uploaded document vector store
uploaded_vector_store = None

# Directory for storing conversation histories
CONVERSATION_DIR = os.path.join(os.path.dirname(__file__), "conversations")
if not os.path.exists(CONVERSATION_DIR):
    os.makedirs(CONVERSATION_DIR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global uploaded_vector_store
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    if file and file.filename.lower().endswith(('.pdf', '.docx')):
        try:
            if file.filename.lower().endswith('.pdf'):
                text_chunks = get_pdf_text(file)
            else:
                text_chunks = get_docx_text(file)
            uploaded_vector_store = create_vector_store(text_chunks)
            return jsonify({"message": "File processed successfully"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Invalid file type. Only PDF and DOCX files are supported"}), 400

def format_model_output(text):
    """
    Minimal post-processing for general chat:
    - Remove consecutive duplicate lines (caused by streaming bugs)
    - Remove excessive blank lines
    - Do not add bold, bullets, or extra formatting
    """
    import re
    lines = text.splitlines()
    cleaned = []
    prev = None
    for line in lines:
        line = line.strip()
        if not line:
            if cleaned and cleaned[-1] == '':
                continue  # skip extra blank lines
            cleaned.append('')
            prev = ''
            continue
        if line == prev:
            continue  # skip duplicate lines
        cleaned.append(line)
        prev = line
    return '\n'.join(cleaned).strip()

def structure_model_output(text):
    """
    Enhanced formatter for model outputs with improved readability:
    - Clearer section headers with distinct formatting
    - Better spacing and organization
    - Enhanced list and equation formatting 
    - Clean paragraph structure with proper indentation
    """
    import re
    
    # Remove duplicate lines and clean up initial spacing
    lines = text.splitlines()
    cleaned = []
    prev = None
    for line in lines:
        line = line.strip()
        if not line:
            if cleaned and cleaned[-1] == '':
                continue
            cleaned.append('')
            prev = ''
            continue
        if line == prev:
            continue
        cleaned.append(line)
        prev = line
    text = '\n'.join(cleaned)

    # Format major section headers (ALL CAPS followed by colon)
    text = re.sub(r'(^|\n)([A-Z][A-Z\s]{2,}:)', r'\1\n\n### \2\n', text)
    
    # Format regular section headers (Capitalized followed by colon)
    text = re.sub(r'(^|\n)([A-Z][a-z\s]{2,}:)', r'\1\n**\2**\n', text)
    
    # Format equations and mathematical expressions
    text = re.sub(r'([A-Za-z0-9_\*\(\)\+\-/=\^ ]{3,}=.+?\^?\d*)', r'`\1`', text)
    
    # Add paragraph breaks after sentences that end sections
    text = re.sub(r'([a-z0-9\)])\. ([A-Z])', r'\1.\n\n\2', text)
    
    # Format bullet points with better indentation and spacing
    text = re.sub(r'(^|\n)[*\-+]\s+', r'\1  • ', text)
    
    # Format numbered lists with consistent indentation
    text = re.sub(r'(^|\n)(\d+)\.\s+', r'\1  \2. ', text)
    
    # Add extra spacing around lists for better readability
    text = re.sub(r'\n((?:  [•\d].*?)(?:\n(?:  [•\d].*?))*)', r'\n\1\n', text)
    
    # Format important terms with emphasis
    text = re.sub(r'\b(NOTE|IMPORTANT|WARNING):', r'**\1:**', text)
    
    # Add spacing around code blocks and equations
    text = re.sub(r'(`[^`]+`)', r'\n\1\n', text)
    
    # Clean up excessive whitespace while maintaining structure
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    text = re.sub(r'^\n+', '', text)
    text = re.sub(r'\n+$', '', text)
    
    return text.strip()

def get_llm_response(prompt):
    """Get response from LLM using direct model interface"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt
            }
        )
        
        if response.status_code == 200:
            return response.json().get('response', '')
        
        print(f"Error response: {response.status_code}")
        return "Error: Could not generate response"
        
    except Exception as e:
        print(f"Error calling LLM: {str(e)}")
        return "Error: Could not connect to model"

def generate_summary(text):
    """Generate a concise summary of the SOP with robust error handling"""
    if not text or not text.strip():
        return "Error: Document appears to be empty"
    
    # Clean up text - remove excessive whitespace and normalize line endings
    text = ' '.join(text.split())
    
    # Take content from beginning, middle and end for better context
    text_length = len(text)
    chunk_size = 2000  # Reduced from 8000 to take from multiple sections
    
    beginning = text[:chunk_size]
    middle_start = max(0, text_length//2 - chunk_size//2)
    middle = text[middle_start:middle_start + chunk_size]
    end = text[max(0, text_length - chunk_size):]
    
    prompt = f"""You are a technical documentation expert. Please analyze this Standard Operating Procedure (SOP) document and provide a clear, structured summary.

Key points to include:
1. Document Purpose: What is this SOP about and what does it aim to achieve?
2. Key Procedures: What are the main steps or processes outlined?
3. Critical Requirements: What are the essential conditions, prerequisites, or safety measures?
4. Target Users: Who should use this document and what roles are involved?

Here are sections from the document for analysis:

BEGINNING:
{beginning}

MIDDLE SECTION:
{middle}

END SECTION:
{end}

Please provide a comprehensive yet concise summary that would help someone quickly understand this SOP's key points. Format the response with clear sections and bullet points where appropriate."""

    try:
        response = get_llm_response(prompt)
        
        # Validate response
        if not response or len(response.strip()) < 50:  # Arbitrary minimum length for a meaningful summary
            return "Error: Could not generate meaningful summary"
            
        # Clean and structure the response
        response = structure_model_output(response)
        
        return response
        
    except Exception as e:
        print(f"Error in generate_summary: {str(e)}")
        return "Error: Failed to generate summary. Please try again."

def generate_mcqs_from_text(text):
    """Generate 10 multiple choice questions from the text"""
    prompt = "Generate 10 multiple choice questions to test understanding of this SOP document. For each question, provide 4 options and indicate the correct answer at the end of 10 mcqs."
    
    response = get_llm_response(prompt + "\n\nSOP Content:\n" + text[:8000])
    return response

def format_conversation_history(history):
    """Format conversation history into a string for the prompt"""
    formatted = ""
    for entry in history[-5:]:  # Only use last 5 exchanges to keep context manageable
        formatted += f"Human: {entry['human']}\nAssistant: {entry['assistant']}\n\n"
    return formatted

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    prompt = data.get('message')
    chat_mode = data.get('chat_mode', 'general')
    
    # Get or create conversation ID
    conversation_id = session.get('conversation_id')
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
        session['conversation_id'] = conversation_id
    
    # Load conversation history
    conversation_file = os.path.join(CONVERSATION_DIR, f"{conversation_id}.json")
    try:
        with open(conversation_file, 'r') as f:
            conversation_history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        conversation_history = []

    try:
        # Format conversation history
        conv_context = format_conversation_history(conversation_history)
        
        if chat_mode == 'preloaded':
            current_sop_id = session.get('current_sop_id')
            if not current_sop_id or current_sop_id not in sop_vector_stores:
                return jsonify({"error": "No SOP selected. Please select an SOP first."}), 400
                
            # Get the current SOP path
            pdf_path = session.get('current_sop_path')
            if not pdf_path or not os.path.exists(pdf_path):
                return jsonify({"error": "SOP file not found"}), 400
                
            # If asking for MCQs, use the full document text
            if "generate 10 mcq" in prompt.lower():
                doc = fitz.open(pdf_path)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                full_prompt = f"""Previous conversation:\n{conv_context}\nPlease help me {prompt}. Use this SOP content:\n{text[:8000]}"""
            else:
                # For regular questions, use vector search
                docs = sop_vector_stores[current_sop_id].similarity_search(prompt, k=3)
                context = "\n".join([doc.page_content for doc in docs])
                full_prompt = f"""Previous conversation:\n{conv_context}\nBased on this SOP content, please answer the following question:\n\nContext:\n{context}\n\nQuestion: {prompt}"""
            
        elif chat_mode == 'upload':
            if uploaded_vector_store is None:
                return jsonify({"error": "No uploaded document available"}), 400
            docs = uploaded_vector_store.similarity_search(prompt, k=3)
            context = "\n".join([doc.page_content for doc in docs])
            full_prompt = f"""Previous conversation:\n{conv_context}\nBased on this document content, please answer the following question:\n\nContext:\n{context}\n\nQuestion: {prompt}"""
        else:  # 'general' mode
            full_prompt = f"""Previous conversation:\n{conv_context}\nHuman: {prompt}\nAssistant: Let me help you with that."""

        # Add system instruction for better context handling
        system_prompt = "You are a helpful AI assistant."
        full_prompt = f"{system_prompt}\n\n{full_prompt}"

        # Request to model API with streaming enabled
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": full_prompt,
                "stream": True
            },
            stream=True
        )
        response.encoding = 'utf-8'

        def generate():
            full_response = ""
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        data = json.loads(line)
                        if 'response' in data:
                            chunk = data['response']
                            full_response += chunk
                            yield f"data:{json.dumps({'response': chunk})}\n\n"
                    except json.JSONDecodeError:
                        continue
            
            # Save to conversation history
            formatted_full = structure_model_output(full_response)
            conversation_history.append({
                'human': prompt,
                'assistant': formatted_full
            })
            with open(conversation_file, 'w') as f:
                json.dump(conversation_history, f)
            yield "data: [DONE]\n\n"
            
        return Response(generate(), mimetype='text/event-stream')

    except Exception as e:
        print(f"Chat error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/clear_conversation', methods=['POST'])
def clear_conversation():
    conversation_id = session.get('conversation_id')
    if conversation_id:
        conversation_file = os.path.join(CONVERSATION_DIR, f"{conversation_id}.json")
        try:
            os.remove(conversation_file)
        except FileNotFoundError:
            pass
        session.pop('conversation_id', None)
    return jsonify({"message": "Conversation cleared"})

@app.route('/generate_mcqs')
def generate_mcqs():
    """Generate MCQs from the currently selected SOP"""
    try:
        # Get current SOP path from session
        pdf_path = session.get('current_sop_path')
        if not pdf_path or not os.path.exists(pdf_path):
            return jsonify({'error': 'No SOP selected. Please select an SOP first.'}), 400
        
        # Read the PDF content
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        
        # Generate MCQs using the LLM
        mcqs = generate_mcqs_from_text(text)
        return jsonify(mcqs)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)






