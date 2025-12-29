import os
import pandas as pd
import numpy as np
import faiss
import gradio as gr
import requests
from sentence_transformers import SentenceTransformer
from groq import Groq

# =========================
# 🔐 Configuration
# =========================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
CSV_URL = "https://drive.google.com/uc?id=1izLDUYFNE3hx6Q9UM7VTEu6qnkFVqUdq"

# =========================
# 🧠 RAG System
# =========================
class RAGSystem:
    def __init__(self):
        self.embedder = None
        self.client = None
        self.index = None
        self.chunks = []
        self.df = None
        self.ready = False
        
    def initialize(self):
        """Initialize RAG system"""
        print("🚀 Starting RAG System...")
        
        # Initialize models
        try:
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            if GROQ_API_KEY:
                self.client = Groq(api_key=GROQ_API_KEY)
                print("✅ Groq client initialized")
            else:
                print("⚠️ No API key - AI disabled")
        except Exception as e:
            print(f"❌ Model init error: {e}")
            return False
        
        # Load CSV
        if not self.load_csv():
            return False
        
        # Create chunks
        self.create_chunks()
        
        # Create index
        if self.chunks:
            self.create_index()
        
        self.ready = True
        print(f"✅ RAG Ready: {len(self.chunks)} chunks from {self.df.shape[0]} rows")
        return True
    
    def load_csv(self):
        """Load CSV from Google Drive"""
        try:
            print("📥 Loading CSV...")
            session = requests.Session()
            response = session.get(CSV_URL, timeout=30)
            response.raise_for_status()
            
            # Try different encodings
            content = response.content
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(pd.io.common.BytesIO(content), encoding=encoding)
                    print(f"✅ Loaded with {encoding} encoding")
                    break
                except:
                    continue
            
            if self.df is None:
                self.df = pd.read_csv(pd.io.common.BytesIO(content), on_bad_lines='skip')
                print("✅ Loaded with error skipping")
            
            print(f"📊 Data: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return True
            
        except Exception as e:
            print(f"❌ CSV error: {e}")
            return False
    
    def create_chunks(self):
        """Create text chunks from dataframe"""
        self.chunks = []
        
        # Add dataset overview
        cols = ", ".join(self.df.columns.tolist()[:10])
        if len(self.df.columns) > 10:
            cols = cols + f"... (+{len(self.df.columns)-10} more)"
        
        overview = f"Dataset: {self.df.shape[0]} rows, {self.df.shape[1]} columns. Columns: {cols}."
        self.chunks.append(overview)
        
        # Add column info (limit to 20 columns)
        for col in self.df.columns[:20]:
            dtype = str(self.df[col].dtype)
            unique_count = self.df[col].nunique()
            
            # Get sample value
            sample_val = ""
            if len(self.df) > 0:
                val = self.df[col].iloc[0]
                if pd.notna(val):
                    sample_val = str(val)[:50]
            
            chunk = f"Column '{col}': Type {dtype}, {unique_count} unique values. Sample: {sample_val}"
            self.chunks.append(chunk)
        
        # Add sample rows (limit to 20 rows)
        for i in range(min(20, len(self.df))):
            row_data = []
            for col in self.df.columns[:5]:  # First 5 columns
                val = self.df.iloc[i][col]
                if pd.notna(val):
                    val_str = str(val)[:50]
                    row_data.append(f"{col}: {val_str}")
            
            if row_data:
                chunk = f"Row {i+1}: {' | '.join(row_data)}"
                self.chunks.append(chunk)
        
        print(f"📝 Created {len(self.chunks)} chunks")
    
    def create_index(self):
        """Create FAISS index"""
        try:
            embeddings = self.embedder.encode(self.chunks)
            embeddings = np.array(embeddings).astype("float32")
            
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(embeddings)
            print(f"✅ Index created: {dim} dimensions")
        except Exception as e:
            print(f"⚠️ Index error: {e}")
    
    def search(self, query, k=5):
        """Search for relevant chunks"""
        if not self.index or not self.chunks:
            return []
        
        try:
            query_embed = self.embedder.encode([query]).astype("float32")
            distances, indices = self.index.search(query_embed, k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.chunks):
                    relevance = 1.0 / (1.0 + distances[0][i])
                    results.append({
                        'text': self.chunks[idx],
                        'relevance': relevance,
                        'index': idx
                    })
            
            # Sort by relevance
            results.sort(key=lambda x: x['relevance'], reverse=True)
            return results[:3]  # Return top 3
            
        except Exception as e:
            print(f"⚠️ Search error: {e}")
            return []
    
    def answer(self, question):
        """Generate answer using RAG"""
        if not question.strip():
            return "Please enter a question.", []
        
        # Get context
        context_results = self.search(question)
        
        if not context_results:
            return "No relevant data found.", []
        
        # Format context for display
        display_contexts = []
        for r in context_results:
            display_contexts.append(r)
        
        # If no AI, return context
        if not self.client:
            context_text = "\n\n".join([f"[Relevance: {r['relevance']:.2f}]\n{r['text']}" 
                                      for r in context_results])
            return context_text, display_contexts
        
        # AI answer
        try:
            # Build context string
            context_text = "\n\n".join([r['text'] for r in context_results])
            
            prompt = f"""Based on this CSV data:
{context_text}
Question: {question}
Answer based ONLY on the context. If unsure, say so."""
            
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful data analyst."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            return answer, display_contexts
            
        except Exception as e:
            return f"AI Error: {str(e)}", display_contexts

# =========================
# 🚀 Initialize
# =========================
rag = RAGSystem()
rag.initialize()

# =========================
# 🎨 Gradio Interface
# =========================
def create_interface():
    custom_css = """
    <style>
    .gradio-container { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        padding: 20px; 
        min-height: 100vh;
    }
    .main-box { 
        max-width: 900px; 
        margin: auto; 
        background: white; 
        border-radius: 20px; 
        padding: 30px; 
        box-shadow: 0 20px 60px rgba(0,0,0,0.2); 
    }
    .header { 
        text-align: center; 
        margin-bottom: 30px; 
    }
    .title { 
        font-size: 32px; 
        font-weight: 800; 
        background: linear-gradient(90deg, #667eea, #764ba2); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        margin-bottom: 10px;
    }
    .status { 
        padding: 15px; 
        border-radius: 10px; 
        text-align: center; 
        margin-bottom: 20px; 
        color: white; 
    }
    .ready { 
        background: linear-gradient(135deg, #00b09b, #96c93d); 
    }
    .error { 
        background: linear-gradient(135deg, #ff6b6b, #ee5a52); 
    }
    .btn-primary { 
        background: linear-gradient(135deg, #ff6ec4 0%, #7873f5 100%); 
        color: white; 
        border: none; 
        padding: 15px; 
        border-radius: 10px; 
        width: 100%; 
        font-weight: 600; 
        cursor: pointer;
    }
    .answer-box { 
        background: #f8f9fa; 
        padding: 20px; 
        border-radius: 10px; 
        margin-top: 20px; 
        border-left: 5px solid #00b09b; 
    }
    /* ANSWER TEXT COLOR STYLES */
    .answer-box strong {
        color: #2d3748;
        font-size: 18px;
        display: block;
        margin-bottom: 10px;
    }
    .answer-content {
    background: linear-gradient(90deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 16px;
    line-height: 1.6;
    margin-top: 10px;
}
}
    .context-box { 
        background: #f1f5f9; 
        padding: 15px; 
        border-radius: 8px; 
        margin-top: 10px; 
        border-left: 3px solid #667eea; 
    }
    .context-box small {
        color: #718096;
        font-weight: bold;
    }
    </style>
    """
    
    with gr.Blocks(title="Simple RAG CSV") as demo:
        # Add CSS
        gr.HTML(custom_css)
        
        with gr.Column(elem_classes="main-box"):
            # Header
            with gr.Column(elem_classes="header"):
                gr.Markdown('<div class="title">📊 Simple RAG CSV</div>')
                gr.Markdown('Ask questions about your CSV data')
            
            # Status
            status_class = "ready" if rag.ready else "error"
            status_text = "✅ RAG System Ready" if rag.ready else "❌ System Error"
            if rag.df is not None:
                status_text = status_text + f" • {rag.df.shape[0]} rows, {rag.df.shape[1]} columns"
            
            gr.Markdown(f'<div class="status {status_class}">{status_text}</div>')
            
            # Input
            question = gr.Textbox(
                label="💬 Your Question",
                placeholder="Example: What columns are in the CSV? Show me sample data.",
                lines=3
            )
            
            with gr.Row():
                submit = gr.Button("🚀 Get Answer", variant="primary", elem_classes="btn-primary")
                clear = gr.Button("🗑️ Clear", variant="secondary")
            
            # Output
            answer = gr.HTML(label="🤖 Answer")
            context = gr.HTML(label="🔍 Retrieved Context")
            
            # Examples
            with gr.Accordion("💡 Example Questions", open=True):
                gr.Examples(
                    examples=[
                        ["What columns are in this CSV?"],
                        ["Show me sample data"],
                        ["How many rows and columns?"],
                        ["What is in the first row?"],
                        ["Describe the dataset"]
                    ],
                    inputs=[question],
                    label="Click to try:"
                )
        
        # Functions
        def process(question_text):
            if not question_text or not question_text.strip():
                empty_html = '''
                <div style="text-align: center; color: #666; padding: 20px;">
                    Please enter a question
                </div>
                '''
                return empty_html, ""
            
            answer_text, contexts = rag.answer(question_text)
            
            # Format answer
            answer_html = f"""
            <div class="answer-box">
                <strong>Answer:</strong>
                <div class="answer-content">
                    {answer_text.replace(chr(10), '<br>')}
                </div>
            </div>
            """
            
            # Format context
            context_html = ""
            if contexts:
                context_html = "<div><strong>Top Contexts:</strong></div>"
                for ctx in contexts:
                    text_display = ctx['text'].replace(chr(10), ' ')
                    context_html = context_html + f"""
                    <div class="context-box">
                        <small>Relevance: {ctx['relevance']:.2f}</small><br>
                        {text_display[:150]}...
                    </div>
                    """
            
            return answer_html, context_html
        
        def clear_all():
            empty_answer = '''
            <div style="text-align: center; color: #666; padding: 20px;">
                Answer will appear here
            </div>
            '''
            empty_context = '''
            <div style="text-align: center; color: #999; font-style: italic; padding: 10px;">
                Context will appear here
            </div>
            '''
            return "", empty_answer, empty_context
        
        # Event handlers
        submit.click(process, inputs=[question], outputs=[answer, context])
        question.submit(process, inputs=[question], outputs=[answer, context])
        clear.click(clear_all, inputs=[], outputs=[question, answer, context])
    
    return demo

# =========================
# 🚀 Launch App
# =========================
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False
    )
