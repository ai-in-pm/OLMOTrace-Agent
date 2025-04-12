# OLMOTRACE Interactive Demo

This repository contains an interactive demonstration of OLMOTRACE, a system that traces large language model outputs back to their training data in real-time.

The development of this repository was inspired by the paper "OLMOTRACE: Tracing Language Model Outputs Back to Trillions of Training Tokens". To read the full article, visit https://arxiv.org/pdf/2504.07096.

![OMOTRACE Interface](https://github.com/user-attachments/assets/10aa712e-38b6-4a97-81df-6f4bcf75b128)
![OMOTRACE Interface II](https://github.com/user-attachments/assets/a43c7464-2ec0-49e3-b5a5-827a70b45cb6)

## About OLMOTRACE

OLMOTRACE is the first system that traces the outputs of language models back to their full, multi-trillion-token training data in real time. It finds and shows verbatim matches between segments of language model output and documents in the training text corpora.

## Demo Features

This interactive demo showcases:

1. How OLMOTRACE identifies verbatim matches in language model outputs
2. The ability to explore source documents from training data
3. Different use cases including fact checking, tracing creative expressions, and math capabilities
4. An interactive interface with two AI agents demonstrating the system
5. **File Upload Functionality**: Upload your own files (PDF, TXT, DOCX, CSV, XLSX) for the AI agents to analyze using OLMOTRACE

## Running the Demo

To run the demo:

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run olmo_agent_demo.py
   ```

3. Open your browser to the URL displayed in the terminal (typically http://localhost:8501)

## Demo Structure

The demo features two simulated AI agents:

- **OLMO Agent 1**: A PhD-level expert who explains OLMOTRACE
- **OLMO Agent 2**: An assistant who demonstrates OLMOTRACE functionality

Both agents simulate using the o3-mini-high LLM and demonstrate OLMOTRACE through interactive examples.

## File Upload Functionality

The demo includes the ability to upload your own files for analysis:

1. Supported file types: PDF, TXT, DOCX, CSV, and XLSX
2. The AI agents will analyze the uploaded file content using OLMOTRACE
3. The system will identify verbatim matches between the file content and the training data
4. Results are displayed with highlighted spans and source documents
5. The agents provide expert commentary on the findings

This feature allows end users to trace any text back to potential sources in the training data, which is useful for research verification, plagiarism detection, and understanding information provenance.

## Academic Summary

For a comprehensive academic summary of OLMOTRACE, including key takeaways and identified issues, please refer to the `academic_summary.md` file in this repository.
