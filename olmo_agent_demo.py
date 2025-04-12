import streamlit as st
from typing import List, Dict, Tuple, Optional
import base64
import io
import re

# Flag to track if PDF processing is available
PDF_SUPPORT = False

# Try to import PDF processing library
try:
    import fitz  # PyMuPDF for PDF processing
    PDF_SUPPORT = True
except ImportError:
    pass  # We'll handle this in the file processing function

# Try to import other file processing libraries
try:
    import docx2txt  # For DOCX processing
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

try:
    import pandas as pd  # For CSV/Excel processing
    PANDAS_SUPPORT = True
except ImportError:
    PANDAS_SUPPORT = False

# Set page configuration
st.set_page_config(
    page_title="OLMOTRACE Interactive Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .agent-message {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #e6f3ff;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .highlight-span {
        background-color: rgba(255, 165, 0, 0.3);
        padding: 2px;
        border-radius: 3px;
    }
    .highlight-span-high {
        background-color: rgba(255, 165, 0, 0.7);
        padding: 2px;
        border-radius: 3px;
    }
    .highlight-span-medium {
        background-color: rgba(255, 165, 0, 0.4);
        padding: 2px;
        border-radius: 3px;
    }
    .highlight-span-low {
        background-color: rgba(255, 165, 0, 0.2);
        padding: 2px;
        border-radius: 3px;
    }
    .document-card {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .document-high {
        border-left: 5px solid rgba(76, 175, 80, 0.8);
    }
    .document-medium {
        border-left: 5px solid rgba(76, 175, 80, 0.5);
    }
    .document-low {
        border-left: 5px solid rgba(76, 175, 80, 0.3);
    }
    .typing-animation::after {
        content: '|';
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0; }
        100% { opacity: 1; }
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_typing' not in st.session_state:
    st.session_state.current_typing = ""
if 'is_typing' not in st.session_state:
    st.session_state.is_typing = False
if 'current_agent' not in st.session_state:
    st.session_state.current_agent = None
if 'demo_stage' not in st.session_state:
    st.session_state.demo_stage = 0
if 'traced_spans' not in st.session_state:
    st.session_state.traced_spans = []
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'selected_span' not in st.session_state:
    st.session_state.selected_span = None
if 'selected_document' not in st.session_state:
    st.session_state.selected_document = None
if 'show_trace_results' not in st.session_state:
    st.session_state.show_trace_results = False
if 'uploaded_file_content' not in st.session_state:
    st.session_state.uploaded_file_content = None
if 'file_analysis_complete' not in st.session_state:
    st.session_state.file_analysis_complete = False

# Mock data for the demo
class MockDocument:
    def __init__(self, title: str, content: str, source: str, relevance: str):
        self.title = title
        self.content = content
        self.source = source
        self.relevance = relevance  # "high", "medium", or "low"

class MockSpan:
    def __init__(self, text: str, start_pos: int, end_pos: int, relevance: str):
        self.text = text
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.relevance = relevance  # "high", "medium", or "low"
        self.documents = []  # List of documents containing this span

# Sample documents for the demo
sample_documents = [
    MockDocument(
        "Space Needle - Wikipedia",
        "The Space Needle was built for the 1962 World Fair, which drew over 2.3 million visitors. Nearly 20,000 people a day used its elevators during the event.",
        "https://en.wikipedia.org/wiki/Space_Needle",
        "high"
    ),
    MockDocument(
        "Seattle Tourism Guide",
        "When visiting Seattle, don't miss the iconic Space Needle. The Space Needle was built for the 1962 World Fair and has been a symbol of the city ever since.",
        "https://seattletourism.com/attractions",
        "medium"
    ),
    MockDocument(
        "History of World Fairs",
        "The 1962 World Fair in Seattle featured the Space Needle as its centerpiece. Other notable World Fairs include the 1889 Paris Exposition which introduced the Eiffel Tower.",
        "https://worldfairs.org/history",
        "low"
    ),
    MockDocument(
        "Mathematical Formulas - Combinatorics",
        "The binomial coefficient formula is: binom{n}{k} = frac{n!}{k!(n-k)!}. For example, binom{10}{4} = frac{10!}{4!(10-4)!} = 210.",
        "https://mathworld.wolfram.com/BinomialCoefficient.html",
        "high"
    ),
    MockDocument(
        "The Hobbit Fan Fiction",
        "Bilbo looked at Gandalf with excitement in his eyes. 'I'm going on an adventure!' he exclaimed, clutching his small bag of belongings.",
        "https://fanfiction.net/hobbit-adventures",
        "medium"
    )
]

# Function to extract text from various file types
def extract_text_from_file(uploaded_file):
    file_type = uploaded_file.type
    file_content = ""

    try:
        # Process PDF files
        if file_type == "application/pdf":
            if PDF_SUPPORT:
                try:
                    with io.BytesIO(uploaded_file.getvalue()) as pdf_file:
                        pdf_document = fitz.open(stream=pdf_file, filetype="pdf")
                        for page_num in range(len(pdf_document)):
                            page = pdf_document.load_page(page_num)
                            file_content += page.get_text()
                except Exception as e:
                    file_content = f"[Error processing PDF: {str(e)}]"
            else:
                file_content = "[PDF processing is not available. Please install PyMuPDF (pip install pymupdf) to enable PDF support.]"

        # Process Word documents
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            if DOCX_SUPPORT:
                try:
                    file_content = docx2txt.process(io.BytesIO(uploaded_file.getvalue()))
                except Exception as e:
                    file_content = f"[Error processing DOCX: {str(e)}]"
            else:
                file_content = "[DOCX processing is not available. Please install docx2txt (pip install docx2txt) to enable DOCX support.]"

        # Process CSV files
        elif file_type == "text/csv":
            if PANDAS_SUPPORT:
                try:
                    df = pd.read_csv(uploaded_file)
                    file_content = df.to_string()
                except Exception as e:
                    file_content = f"[Error processing CSV: {str(e)}]"
            else:
                file_content = "[CSV processing is not available. Please install pandas (pip install pandas) to enable CSV support.]"

        # Process Excel files
        elif file_type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
            if PANDAS_SUPPORT:
                try:
                    df = pd.read_excel(uploaded_file)
                    file_content = df.to_string()
                except Exception as e:
                    file_content = f"[Error processing Excel: {str(e)}]"
            else:
                file_content = "[Excel processing is not available. Please install pandas and openpyxl (pip install pandas openpyxl) to enable Excel support.]"

        # Process plain text files - this should always work
        elif file_type == "text/plain":
            file_content = uploaded_file.getvalue().decode("utf-8")

        # Default case for other file types - try to decode as text
        else:
            try:
                file_content = uploaded_file.getvalue().decode("utf-8")
            except UnicodeDecodeError:
                file_content = "[File content could not be decoded. Unsupported file type.]"

    except Exception as e:
        file_content = f"[Error processing file: {str(e)}]"

    # Fallback to basic text extraction if file_content is empty or contains error message
    if not file_content or file_content.startswith("[Error") or file_content.startswith("[PDF") or file_content.startswith("[DOCX") or file_content.startswith("[CSV") or file_content.startswith("[Excel"):
        try:
            # Try simple text extraction as fallback
            file_content = uploaded_file.getvalue().decode("utf-8", errors="replace")
            if not file_content.strip():
                file_content = "[Unable to extract readable text from this file. Please try a different file format.]"
        except Exception:
            # If all else fails, at least return something
            if not file_content:
                file_content = "[Unable to extract content from this file. Please try a different file format.]"

    return file_content

# Function to add messages directly without typing simulation
def simulate_typing(agent_name: str, message: str):
    # Check if this exact message from this agent already exists to prevent duplicates
    if not st.session_state.messages or not any(msg["role"] == agent_name and msg["content"] == message for msg in st.session_state.messages):
        # Add the message directly to the conversation history
        st.session_state.messages.append({"role": agent_name, "content": message})
        st.rerun()

# Function to highlight spans in text
def highlight_spans(text: str, spans: List[MockSpan], selected_span: Optional[MockSpan] = None) -> str:
    # Sort spans by start position in reverse order to avoid index shifting
    sorted_spans = sorted(spans, key=lambda x: x.start_pos, reverse=True)

    # If a span is selected, only highlight that one
    if selected_span:
        for span in sorted_spans:
            if span.text == selected_span.text:
                text = text[:span.start_pos] + f'<span class="highlight-span-{span.relevance}">{text[span.start_pos:span.end_pos]}</span>' + text[span.end_pos:]
        return text

    # Otherwise highlight all spans
    for span in sorted_spans:
        text = text[:span.start_pos] + f'<span class="highlight-span-{span.relevance}">{text[span.start_pos:span.end_pos]}</span>' + text[span.end_pos:]

    return text

# Function to trace LM output or uploaded file content
def trace_output(content: str, is_uploaded_file: bool = False) -> Tuple[List[MockSpan], List[MockDocument]]:
    # For demo purposes, we'll create some mock spans and link them to our sample documents
    spans = []

    # Space Needle span
    if "Space Needle" in content and "1962 World Fair" in content:
        start_pos = content.find("The Space Needle was built for the 1962 World Fair")
        if start_pos >= 0:
            span_text = "The Space Needle was built for the 1962 World Fair"
            span = MockSpan(span_text, start_pos, start_pos + len(span_text), "high")
            span.documents = [sample_documents[0], sample_documents[1]]
            spans.append(span)

    # Adventure span
    if "going on an adventure" in content:
        start_pos = content.find("I'm going on an adventure")
        if start_pos >= 0:
            span_text = "I'm going on an adventure"
            span = MockSpan(span_text, start_pos, start_pos + len(span_text), "medium")
            span.documents = [sample_documents[4]]
            spans.append(span)

    # Math span
    if "binom{10}{4}" in content:
        start_pos = content.find("binom{10}{4} = frac{10!}{4!(10-4)!} = 210")
        if start_pos >= 0:
            span_text = "binom{10}{4} = frac{10!}{4!(10-4)!} = 210"
            span = MockSpan(span_text, start_pos, start_pos + len(span_text), "high")
            span.documents = [sample_documents[3]]
            spans.append(span)

    # Additional patterns for uploaded files
    if is_uploaded_file:
        # Look for common patterns in academic papers
        paper_title_pattern = re.compile(r'([A-Z][\w\s:]+)\s*Abstract')
        matches = paper_title_pattern.findall(content)
        for match in matches[:1]:  # Just use the first match for demo purposes
            start_pos = content.find(match)
            if start_pos >= 0:
                span_text = match.strip()
                span = MockSpan(span_text, start_pos, start_pos + len(span_text), "high")
                # Create a mock document for this title
                mock_doc = MockDocument(
                    "Academic Paper Database",
                    f"The paper titled '{span_text}' appears in our academic database with similar research focus.",
                    "https://academic-papers-database.org/search",
                    "high"
                )
                span.documents = [mock_doc]
                spans.append(span)

        # Look for citations
        citation_pattern = re.compile(r'\(([A-Za-z]+\s+et\s+al\.?,\s+\d{4})\)')
        matches = citation_pattern.findall(content)
        for match in matches[:3]:  # Limit to 3 citations for demo
            start_pos = content.find(match)
            if start_pos >= 0:
                span_text = match
                span = MockSpan(span_text, start_pos - 1, start_pos + len(span_text) + 1, "medium")
                # Create a mock document for this citation
                mock_doc = MockDocument(
                    f"Citation: {match}",
                    f"This citation refers to research by {match} which discusses similar concepts.",
                    "https://citation-database.org/lookup",
                    "medium"
                )
                span.documents = [mock_doc]
                spans.append(span)

    # Get all unique documents referenced by spans
    documents = []
    for span in spans:
        for doc in span.documents:
            if doc not in documents:
                documents.append(doc)

    # Sort documents by relevance
    documents.sort(key=lambda x: {"high": 0, "medium": 1, "low": 2}[x.relevance])

    return spans, documents

# Function to run the demo
def run_demo():
    # Title and introduction
    st.title("OLMOTRACE Interactive Demo")

    # Sidebar with information
    with st.sidebar:
        st.header("About OLMOTRACE")
        st.write("""
        OLMOTRACE is a system that traces language model outputs back to their training data in real-time.

        This demo showcases how OLMOTRACE works through an interactive conversation between two AI agents:

        - **OLMO Agent 1**: PhD-level expert who explains OLMOTRACE
        - **OLMO Agent 2**: Assistant who demonstrates OLMOTRACE functionality

        Both agents are simulated to use the o3-mini-high LLM.
        """)

        # File upload section in sidebar
        st.header("Upload Your Own File")
        st.write("Upload a file to have the AI Agents analyze it using OLMOTRACE.")

        # Check if file analysis is already complete
        _ = st.session_state.get('file_analysis_complete', False)  # Just to access the session state

        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "docx", "csv", "xlsx"], key="file_uploader")

        if uploaded_file is not None and not st.session_state.file_analysis_complete:
            st.info("File uploaded. Processing...")
            # Extract text from the uploaded file
            file_content = extract_text_from_file(uploaded_file)
            st.session_state.uploaded_file_content = file_content

            # Trace the file content
            spans, documents = trace_output(file_content, is_uploaded_file=True)
            if spans:
                st.session_state.traced_spans = spans
                st.session_state.documents = documents
                st.session_state.show_trace_results = True
                st.session_state.file_analysis_complete = True

                # Clear any existing messages to prevent duplicates when analyzing a file
                if len(st.session_state.messages) > 0:
                    st.session_state.messages = []

                st.success("File processed successfully! The AI Agents will now analyze it.")
                st.experimental_rerun()  # Rerun once to update the UI
            else:
                st.warning("No traceable content found in the file. Try another file or continue with the demo.")

        st.header("Demo Controls")
        if st.button("Reset Demo"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

    # Main chat area
    chat_col, trace_col = st.columns([3, 2])

    with chat_col:
        st.header("Interactive Demonstration")

        # Display chat history
        for message in st.session_state.messages:
            if message["role"] == "User":
                st.markdown(f'<div class="user-message"><strong>End User:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            elif message["role"] == "OLMO Agent 1":
                st.markdown(f'<div class="agent-message"><strong>OLMO Agent 1 (PhD Expert):</strong> {message["content"]}</div>', unsafe_allow_html=True)
            elif message["role"] == "OLMO Agent 2":
                # If this message has traced spans, highlight them
                content = message["content"]
                if hasattr(message, 'spans') and message['spans']:
                    content = highlight_spans(content, message['spans'], st.session_state.selected_span)
                st.markdown(f'<div class="agent-message"><strong>OLMO Agent 2 (Demo Assistant):</strong> {content}</div>', unsafe_allow_html=True)

        # No typing animation in this version

    # Trace results area (only shown when tracing is active)
    with trace_col:
        if st.session_state.show_trace_results:
            st.header("OLMOTRACE Results")

            # Display traced spans and documents
            if st.session_state.traced_spans:
                # If a document is selected, show only spans from that document
                if st.session_state.selected_document:
                    st.subheader(f"Spans in selected document")
                    doc = st.session_state.selected_document
                    for span in st.session_state.traced_spans:
                        if doc in span.documents:
                            if st.button(f"🔍 {span.text}", key=f"span_{span.text}"):
                                st.session_state.selected_span = span
                                st.session_state.selected_document = None
                                st.rerun()

                    st.markdown("---")
                    st.subheader("Selected Document")
                    st.markdown(f'<div class="document-card document-{doc.relevance}"><strong>{doc.title}</strong><br>{doc.content}<br><small>Source: {doc.source}</small></div>', unsafe_allow_html=True)

                    if st.button("Clear Selection", key="clear_doc"):
                        st.session_state.selected_document = None
                        st.session_state.selected_span = None
                        st.rerun()

                # If a span is selected, show documents containing that span
                elif st.session_state.selected_span:
                    st.subheader(f"Documents containing selected span")
                    span = st.session_state.selected_span
                    st.markdown(f'<div class="highlight-span-{span.relevance}">{span.text}</div>', unsafe_allow_html=True)

                    for doc in span.documents:
                        if st.button(f"📄 {doc.title}", key=f"doc_{doc.title}"):
                            st.session_state.selected_document = doc
                            st.rerun()

                    if st.button("Clear Selection", key="clear_span"):
                        st.session_state.selected_span = None
                        st.rerun()

                # Otherwise show all spans and documents
                else:
                    st.subheader("Highlighted Spans")
                    for span in st.session_state.traced_spans:
                        if st.button(f"🔍 {span.text}", key=f"span_{span.text}"):
                            st.session_state.selected_span = span
                            st.rerun()

                    st.markdown("---")
                    st.subheader("Training Documents")
                    for doc in st.session_state.documents:
                        if st.button(f"📄 {doc.title}", key=f"doc_{doc.title}"):
                            st.session_state.selected_document = doc
                            st.rerun()
            else:
                st.info("No traces found in the current response.")

    # Check if a file has been uploaded and analyzed
    if st.session_state.uploaded_file_content is not None and st.session_state.file_analysis_complete:
        # Display file analysis section
        # Check if we've already added the file analysis messages
        file_analysis_started = any("uploaded file" in msg.get("content", "") for msg in st.session_state.messages if msg.get("role") == "OLMO Agent 1")
        file_analysis_completed = any("Excellent analysis" in msg.get("content", "") for msg in st.session_state.messages if msg.get("role") == "OLMO Agent 1")

        if not file_analysis_started:
            # AI Agents analyze the uploaded file - first message
            simulate_typing("OLMO Agent 1", f"I see that an End User has uploaded a file for analysis. OLMO Agent 2, could you please analyze this file using OLMOTRACE and explain what you find?")

            # Truncate file content for display if it's too long
            display_content = st.session_state.uploaded_file_content
            if len(display_content) > 1000:
                display_content = display_content[:1000] + "... [content truncated]"

            simulate_typing("OLMO Agent 2", f"I'd be happy to analyze this uploaded file using OLMOTRACE. Here's the content I'll be tracing:\n\n```\n{display_content}\n```\n\nLet me trace this content back to the training data.")

        # Check if we need to add the analysis results
        analysis_added = any("After analyzing the uploaded file" in msg.get("content", "") for msg in st.session_state.messages if msg.get("role") == "OLMO Agent 2")
        if file_analysis_started and not analysis_added and len(st.session_state.traced_spans) > 0:
            # Create a message with analysis results
            analysis_message = "After analyzing the uploaded file with OLMOTRACE, I've found several interesting connections to the training data:\n\n"

            # Add information about each traced span
            for i, span in enumerate(st.session_state.traced_spans[:5]):
                analysis_message += f"**Match {i+1}**: '{span.text}'\n"
                for doc in span.documents[:2]:
                    analysis_message += f"- Found in: {doc.title}\n- Context: {doc.content}\n\n"

            analysis_message += "These matches show how the content in the uploaded file relates to documents in the training data. This can help verify information sources, identify potential influences, and understand the origins of specific phrases or concepts."

            # Add the analysis to the conversation
            simulate_typing("OLMO Agent 2", analysis_message)

        # Add expert commentary if not already added
        if not file_analysis_completed and analysis_added:
            simulate_typing("OLMO Agent 1", "Excellent analysis! OLMOTRACE has successfully identified several verbatim matches between the uploaded file and the training data. This demonstrates how OLMOTRACE can be used to analyze any text, not just language model outputs.\n\nEnd Users can upload their own documents to trace content back to potential sources, which is useful for research verification, plagiarism detection, and understanding the provenance of information.")

    # Progress the demo based on current stage if no file is uploaded
    elif st.session_state.demo_stage == 0:
        # Introduction by OLMO Agent 1
        simulate_typing("OLMO Agent 1", "Hello! I'm OLMO Agent 1, a PhD-level expert in language model transparency. Today, I'll be guiding OLMO Agent 2 to demonstrate OLMOTRACE, a groundbreaking system that traces language model outputs back to their training data in real-time.")
        simulate_typing("OLMO Agent 1", "OLMO Agent 2, could you please explain what OLMOTRACE is and show us how it works with a practical example?")
        st.session_state.demo_stage = 1

    elif st.session_state.demo_stage == 1:
        # OLMO Agent 2 explains OLMOTRACE
        simulate_typing("OLMO Agent 2", "Thank you, OLMO Agent 1. I'd be happy to demonstrate OLMOTRACE.")
        simulate_typing("OLMO Agent 2", "OLMOTRACE is the first system that traces language model outputs back to their full, multi-trillion-token training data in real time. It finds and displays verbatim matches between segments of language model output and documents in the training text corpora.")
        simulate_typing("OLMO Agent 2", "Let me show you how it works with a practical example. I'll generate a response about a famous landmark in Seattle, and then we'll trace it back to the training data.")

        # Generate a response with traceable content
        lm_response = "Seattle is home to many famous landmarks. The Space Needle was built for the 1962 World Fair, which drew millions of visitors. Standing at 605 feet tall, it offers panoramic views of the city, Puget Sound, and the surrounding mountains. It has become an iconic symbol of Seattle and the Pacific Northwest."

        # Add the response to messages
        st.session_state.messages.append({"role": "OLMO Agent 2", "content": lm_response})

        # Trace the output
        spans, documents = trace_output(lm_response)
        st.session_state.traced_spans = spans
        st.session_state.documents = documents
        st.session_state.show_trace_results = True

        # Add spans to the message for highlighting
        st.session_state.messages[-1]['spans'] = spans

        st.session_state.demo_stage = 2
        st.rerun()

    elif st.session_state.demo_stage == 2:
        # OLMO Agent 1 explains the tracing results
        simulate_typing("OLMO Agent 1", "Excellent demonstration! As you can see, OLMOTRACE has identified a verbatim match in the response: 'The Space Needle was built for the 1962 World Fair'.")
        simulate_typing("OLMO Agent 1", "The system has highlighted this span and shown the source documents from the training data that contain this exact text. The documents are ranked by relevance, with the most relevant ones highlighted more brightly.")
        simulate_typing("OLMO Agent 1", "OLMO Agent 2, could you explain how users can interact with these results and show us another example, perhaps demonstrating a different use case?")
        st.session_state.demo_stage = 3

    elif st.session_state.demo_stage == 3:
        # OLMO Agent 2 explains interaction and shows another example
        simulate_typing("OLMO Agent 2", "Certainly! Users can interact with OLMOTRACE results in several ways:")
        simulate_typing("OLMO Agent 2", "1. Click on a highlighted span to see all documents containing that exact text\n2. Click on a document to see all spans from that document\n3. View the full context of any document to better understand the source")
        simulate_typing("OLMO Agent 2", "Let me demonstrate another use case: tracing 'creative' expressions. I'll generate a story in a Tolkien-inspired style.")

        # Generate a response with creative expression
        lm_response = "The old wizard looked at the small hobbit with amusement in his eyes. 'Where are you off to in such a hurry?' he asked. The hobbit adjusted his backpack and replied with excitement, 'I'm going on an adventure! Beyond the Shire, there are mountains to climb and dragons to face. I may be small, but I have courage in my heart.'"

        # Reset previous traces
        st.session_state.selected_span = None
        st.session_state.selected_document = None

        # Add the response to messages
        st.session_state.messages.append({"role": "OLMO Agent 2", "content": lm_response})

        # Trace the output
        spans, documents = trace_output(lm_response)
        st.session_state.traced_spans = spans
        st.session_state.documents = documents

        # Add spans to the message for highlighting
        st.session_state.messages[-1]['spans'] = spans

        st.session_state.demo_stage = 4
        st.rerun()

    elif st.session_state.demo_stage == 4:
        # OLMO Agent 1 explains the creative expression tracing
        simulate_typing("OLMO Agent 1", "This is fascinating! OLMOTRACE has identified that the phrase 'I'm going on an adventure' appears verbatim in the training data, specifically in what appears to be Hobbit fan fiction.")
        simulate_typing("OLMO Agent 1", "This demonstrates how OLMOTRACE can help us understand when language models are reproducing phrases they've seen during training, even when generating seemingly creative content.")
        simulate_typing("OLMO Agent 1", "OLMO Agent 2, could you show us one final example, perhaps demonstrating how OLMOTRACE can trace mathematical capabilities?")
        st.session_state.demo_stage = 5

    elif st.session_state.demo_stage == 5:
        # OLMO Agent 2 demonstrates math tracing
        simulate_typing("OLMO Agent 2", "Absolutely! Let me demonstrate how OLMOTRACE can trace mathematical capabilities. I'll solve a combinatorial problem.")

        # Generate a response with math content
        lm_response = "To solve this combinatorial problem, we need to calculate the number of ways to select 4 items from a set of 10 items, without regard to order.\n\nThis is given by the binomial coefficient formula:\nbinom{10}{4} = frac{10!}{4!(10-4)!} = 210\n\nTherefore, there are 210 possible combinations."

        # Reset previous traces
        st.session_state.selected_span = None
        st.session_state.selected_document = None

        # Add the response to messages
        st.session_state.messages.append({"role": "OLMO Agent 2", "content": lm_response})

        # Trace the output
        spans, documents = trace_output(lm_response)
        st.session_state.traced_spans = spans
        st.session_state.documents = documents

        # Add spans to the message for highlighting
        st.session_state.messages[-1]['spans'] = spans

        st.session_state.demo_stage = 6
        st.rerun()

    elif st.session_state.demo_stage == 6:
        # OLMO Agent 1 concludes the demonstration
        simulate_typing("OLMO Agent 1", "Excellent! OLMOTRACE has identified that the mathematical calculation 'binom{10}{4} = frac{10!}{4!(10-4)!} = 210' appears verbatim in the training data.")
        simulate_typing("OLMO Agent 1", "This shows how OLMOTRACE can help us understand when language models are reproducing mathematical calculations they've seen during training, which provides insight into how these models learn to perform mathematical operations.")
        simulate_typing("OLMO Agent 1", "To summarize, OLMOTRACE offers three key benefits:")
        simulate_typing("OLMO Agent 1", "1. **Transparency**: It reveals direct connections between model outputs and training data\n2. **Understanding**: It helps users understand model behavior through the lens of training data\n3. **Exploration**: It enables exploration of fact checking, creativity, and capabilities")
        simulate_typing("OLMO Agent 1", "Thank you for joining our demonstration of OLMOTRACE. This system represents an important step toward more transparent and understandable AI systems.")

        # Add a user input for questions
        user_input = st.text_input("Ask a question about OLMOTRACE:")
        if user_input:
            st.session_state.messages.append({"role": "User", "content": user_input})
            simulate_typing("OLMO Agent 1", f"Thank you for your question: '{user_input}'. OLMOTRACE is an open-source system developed by the Allen Institute for AI. It's available through the AI2 Playground and currently supports the OLMo family of models. The system traces language model outputs back to their training data in real-time, completing traces within approximately 4.5 seconds.")
            st.rerun()

# Run the demo
run_demo()
