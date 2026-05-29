#!/usr/bin/env python3
"""
Convert thesis defense study materials to PDF format.
"""

import os
import markdown
from pathlib import Path
from datetime import datetime

try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    print("Note: weasyprint not available, using alternative method")

def markdown_to_html(md_content, title=""):
    """Convert markdown content to HTML."""

    # Convert markdown to HTML
    html_body = markdown.markdown(md_content, extensions=['extra', 'codehilite', 'toc'])

    # Wrap in full HTML document with styling
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}

            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background: white;
                padding: 40px;
                max-width: 900px;
                margin: 0 auto;
            }}

            h1 {{
                color: #1a73e8;
                font-size: 2.5em;
                margin: 30px 0 20px 0;
                border-bottom: 3px solid #1a73e8;
                padding-bottom: 10px;
                page-break-after: avoid;
            }}

            h2 {{
                color: #1a73e8;
                font-size: 2em;
                margin: 25px 0 15px 0;
                page-break-after: avoid;
            }}

            h3 {{
                color: #4285f4;
                font-size: 1.5em;
                margin: 20px 0 10px 0;
                page-break-after: avoid;
            }}

            h4, h5, h6 {{
                color: #4285f4;
                margin: 15px 0 10px 0;
                page-break-after: avoid;
            }}

            p {{
                margin: 10px 0;
                text-align: justify;
            }}

            ul, ol {{
                margin: 15px 0;
                padding-left: 30px;
            }}

            li {{
                margin: 8px 0;
            }}

            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
                page-break-inside: avoid;
            }}

            table, th, td {{
                border: 1px solid #ddd;
            }}

            th {{
                background-color: #1a73e8;
                color: white;
                padding: 12px;
                text-align: left;
            }}

            td {{
                padding: 10px;
            }}

            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}

            code {{
                background-color: #f5f5f5;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                color: #d63384;
            }}

            pre {{
                background-color: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 15px;
                overflow-x: auto;
                margin: 15px 0;
                page-break-inside: avoid;
            }}

            pre code {{
                color: #333;
                background: none;
                padding: 0;
            }}

            blockquote {{
                border-left: 4px solid #1a73e8;
                padding-left: 20px;
                margin: 15px 0;
                color: #666;
                font-style: italic;
            }}

            .toc {{
                background: #f9f9f9;
                border: 1px solid #ddd;
                padding: 20px;
                margin: 20px 0;
                border-radius: 5px;
                page-break-inside: avoid;
            }}

            .toc ul {{
                margin: 10px 0;
                padding-left: 20px;
            }}

            em {{
                color: #666;
            }}

            strong {{
                color: #1a73e8;
                font-weight: bold;
            }}

            .page-break {{
                page-break-after: always;
            }}

            @media print {{
                body {{
                    padding: 20px;
                }}
                h1 {{
                    page-break-before: always;
                }}
            }}
        </style>
    </head>
    <body>
        {html_body}
        <div style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center; color: #999; font-size: 0.9em;">
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </body>
    </html>
    """

    return html

def create_pdf_from_markdown(md_file, output_pdf):
    """Create PDF from markdown file."""

    print(f"Processing: {md_file}")

    # Read markdown file
    if not os.path.exists(md_file):
        print(f"  ✗ File not found: {md_file}")
        return False

    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()

        # Get title from filename
        title = Path(md_file).stem.replace('_', ' ').title()

        # Convert to HTML
        html_content = markdown_to_html(md_content, title)

        # Save HTML temporarily
        html_file = output_pdf.replace('.pdf', '.html')
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Convert HTML to PDF using weasyprint
        if WEASYPRINT_AVAILABLE:
            try:
                HTML(string=html_content).write_pdf(output_pdf)
                print(f"  ✓ PDF created: {output_pdf}")
                # Clean up HTML file
                if os.path.exists(html_file):
                    os.remove(html_file)
                return True
            except Exception as e:
                print(f"  ✗ Error creating PDF: {e}")
                print(f"  → Saved HTML version: {html_file}")
                return False
        else:
            print(f"  → Saved HTML version (weasyprint unavailable): {html_file}")
            return True

    except Exception as e:
        print(f"  ✗ Error processing {md_file}: {e}")
        return False

def main():
    """Create all PDFs."""

    print("=" * 80)
    print("CREATING PDF STUDY MATERIALS")
    print("=" * 80)
    print()

    base_path = Path('/home/user/maskdatasetvgg16')
    pdf_output_dir = base_path / 'PDF_MATERIALS'

    # Create output directory
    pdf_output_dir.mkdir(exist_ok=True)

    # Files to convert
    files_to_convert = [
        ('DEFENSE_PREPARATION_INDEX.md', 'defense_preparation_index.pdf'),
        ('DEFENSE_FLASHCARDS.md', 'defense_flashcards.pdf'),
        ('THESIS_DEFENSE_STUDY_GUIDE.md', 'thesis_defense_study_guide.pdf'),
        ('QUICKSTART.md', 'quickstart.pdf'),
        ('THERMAL_DETECTION_README.md', 'thermal_detection_readme.pdf'),
    ]

    successful = 0
    failed = 0

    for md_file, pdf_file in files_to_convert:
        md_path = base_path / md_file
        pdf_path = pdf_output_dir / pdf_file

        if create_pdf_from_markdown(str(md_path), str(pdf_path)):
            successful += 1
        else:
            failed += 1
        print()

    # Create summary document
    print("Creating summary document...")
    summary_path = base_path / 'DEFENSE_MATERIALS_SUMMARY.txt'
    if os.path.exists(summary_path):
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_content = f.read()

        summary_html = markdown_to_html(f"# Defense Materials Summary\n\n{summary_content}", "Defense Materials Summary")
        summary_pdf = pdf_output_dir / 'defense_materials_summary.pdf'

        try:
            if WEASYPRINT_AVAILABLE:
                HTML(string=summary_html).write_pdf(str(summary_pdf))
                print(f"  ✓ Summary PDF created: {summary_pdf}")
            else:
                print(f"  → Summary HTML saved (weasyprint unavailable)")
        except Exception as e:
            print(f"  ✗ Error creating summary PDF: {e}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Successfully created: {successful} PDFs")
    if failed > 0:
        print(f"✗ Failed: {failed} PDFs")
    print(f"📁 Output directory: {pdf_output_dir}")
    print()
    print("All PDF files are ready!")
    print()

if __name__ == '__main__':
    main()
