import re
from pypdf import PdfReader


class PDFLoader:

    def __init__(self, pdf_path):

        self.pdf_path = pdf_path

    def clean_text(self, text):

        import re

        # Fix broken line wraps inside paragraphs
        text = re.sub(r'(?<!\n)\s*\n\s*(?!\n)', ' ', text)

        # Remove excessive spaces/tabs
        text = re.sub(r'[ \t]+', ' ', text)

        # Normalize excessive blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def load_pdf(self):

        reader = PdfReader(self.pdf_path)

        text = ""

        for page in reader.pages:

            extracted_text = page.extract_text()

            if extracted_text:

                text += extracted_text + "\n"

        cleaned_text = self.clean_text(text)
        print(cleaned_text[:200])
        return cleaned_text