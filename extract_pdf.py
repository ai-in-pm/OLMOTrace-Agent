import PyPDF2
import sys
import io

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
        return text

if __name__ == "__main__":
    pdf_path = "OLMOTRACE PAPER.pdf"
    try:
        text = extract_text_from_pdf(pdf_path)
        with io.open('pdf_content.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        print("Content extracted to pdf_content.txt")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
