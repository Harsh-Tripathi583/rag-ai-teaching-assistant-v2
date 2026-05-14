from pypdf import PdfReader


class PDFLoader:

    def __init__(self, pdf_path):

        self.pdf_path = pdf_path

    def load_pdf(self):

        reader = PdfReader(self.pdf_path)

        full_text = ""

        for page in reader.pages:

            extracted_text = page.extract_text()

            if extracted_text:

                full_text += extracted_text + "\n"

        return full_text