import pathlib, pdfplumber, sys

pdf_path = pathlib.Path('gpay_statement_20260301_20260531.pdf')
if not pdf_path.is_file():
    print('PDF not found', file=sys.stderr)
    sys.exit(1)

output_path = pathlib.Path('gpay_pdf_text.txt')
with pdfplumber.open(pdf_path) as pdf, output_path.open('w', encoding='utf-8') as out:
    for i, page in enumerate(pdf.pages, start=1):
        txt = page.extract_text() or ''
        out.write(f'--- Page {i} ---\n')
        out.write(txt)
        out.write('\n\n')
print(f'Extracted text written to {output_path}')
