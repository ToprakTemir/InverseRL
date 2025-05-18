import fitz  # PyMuPDF
from pathlib import Path

# List your PDF filenames in order (replace with actual paths if needed)
pdf_paths = [
    "comparisons_distance_reward.pdf",
    "comparisons_final_distance.pdf",
    "comparisons_pull_success_0.pdf",
    "comparisons_pull_success_0.01.pdf",
    "comparisons_pull_success_0.05.pdf"
]

# Open each single-page PDF and extract the page
pages = []
for path in pdf_paths:
    doc = fitz.open(path)
    assert len(doc) == 1, f"{path} is not a single-page PDF"
    pages.append(doc.load_page(0))  # Get the first page

# Create a new PDF
output_doc = fitz.open()

# Calculate combined width and max height
total_width = sum(p.rect.width for p in pages)
max_height = max(p.rect.height for p in pages)

# Create a blank page with combined width
new_page = output_doc.new_page(width=total_width, height=max_height)

# Paste pages side by side
x_offset = 0
for p in pages:
    rect = fitz.Rect(
        x_offset,
        0,
        x_offset + p.rect.width,
        p.rect.height
    )
    new_page.show_pdf_page(rect, p.parent, p.number)
    x_offset += p.rect.width

# Save merged PDF
output_doc.save("merged_model_comparisons.pdf")
output_doc.close()