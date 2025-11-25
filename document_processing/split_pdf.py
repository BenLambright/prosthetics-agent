from PyPDF2 import PdfReader, PdfWriter



def split_pdf(input_path, output_prefix="pages/page"):
    reader = PdfReader(input_path)
    for i, page in enumerate(reader.pages):
        writer = PdfWriter()
        writer.add_page(page)
        with open(f"{output_prefix}_{i+1}.pdf", "wb") as f:
            writer.write(f)
    print("Done!")

split_pdf("431195408-Phoenix-v2-assembly-guide-pdf.pdf")
