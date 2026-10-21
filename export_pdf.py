"""
Utility script to export walkthrough.md to high-resolution self-contained HTML and PDF.
Converts all embedded images into Base64 data URIs and uses Microsoft Edge in headless mode.
"""

import os
import re
import base64
import subprocess
import markdown

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(WORKSPACE_DIR, "walkthrough.md")
HTML_PATH = os.path.join(WORKSPACE_DIR, "walkthrough.html")
PDF_PATH = os.path.join(WORKSPACE_DIR, "walkthrough.pdf")
EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

def main():
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_content = f.read()

    def img_replacer(match):
        alt = match.group(1)
        rel_path = match.group(2)
        full_path = os.path.normpath(os.path.join(WORKSPACE_DIR, rel_path))
        if os.path.exists(full_path):
            with open(full_path, "rb") as img_f:
                b64_data = base64.b64encode(img_f.read()).decode("utf-8")
            ext = os.path.splitext(full_path)[1].lower().replace(".", "")
            mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"
            return f'![{alt}](data:{mime};base64,{b64_data})'
        return match.group(0)

    md_content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', img_replacer, md_content)

    html_body = markdown.markdown(
        md_content,
        extensions=['tables', 'fenced_code', 'nl2br', 'sane_lists']
    )

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Topology-Aware Deep Learning Walkthrough</title>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    @page {{
        size: A4;
        margin: 18mm 16mm 18mm 16mm;
        @bottom-right {{
            content: counter(page);
        }}
    }}

    body {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #1e293b;
        background-color: #ffffff;
        line-height: 1.6;
        font-size: 13.5px;
        margin: 0;
        padding: 0;
    }}

    h1 {{
        color: #0f172a;
        font-size: 24px;
        font-weight: 800;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 8px;
        margin-top: 10px;
        margin-bottom: 12px;
    }}

    h2 {{
        color: #1e3a8a;
        font-size: 18px;
        font-weight: 700;
        border-bottom: 1.5px solid #e2e8f0;
        padding-bottom: 6px;
        margin-top: 26px;
        margin-bottom: 12px;
        page-break-after: avoid;
    }}

    h3 {{
        color: #2563eb;
        font-size: 15px;
        font-weight: 600;
        margin-top: 18px;
        margin-bottom: 8px;
        page-break-after: avoid;
    }}

    h4 {{
        color: #334155;
        font-size: 14px;
        font-weight: 600;
        margin-top: 14px;
        margin-bottom: 6px;
        page-break-after: avoid;
    }}

    p {{
        margin: 6px 0 10px 0;
    }}

    blockquote {{
        border-left: 4px solid #3b82f6;
        background-color: #f8fafc;
        padding: 10px 14px;
        margin: 12px 0;
        color: #475569;
        font-size: 13px;
        border-radius: 0 6px 6px 0;
    }}

    table {{
        width: 100%;
        border-collapse: collapse;
        margin: 14px 0 18px 0;
        font-size: 12.5px;
        page-break-inside: avoid;
    }}

    th, td {{
        border: 1px solid #cbd5e1;
        padding: 8px 10px;
        text-align: left;
    }}

    th {{
        background-color: #f1f5f9;
        color: #0f172a;
        font-weight: 600;
    }}

    tr:nth-child(even) {{
        background-color: #f8fafc;
    }}

    pre {{
        background-color: #0f172a;
        color: #e2e8f0;
        padding: 12px 14px;
        border-radius: 6px;
        font-family: 'JetBrains Mono', Consolas, Monaco, monospace;
        font-size: 11.5px;
        line-height: 1.45;
        overflow-x: auto;
        page-break-inside: avoid;
        margin: 10px 0;
    }}

    code {{
        font-family: 'JetBrains Mono', Consolas, Monaco, monospace;
        background-color: #f1f5f9;
        color: #0f172a;
        padding: 2px 5px;
        border-radius: 4px;
        font-size: 12px;
    }}

    pre code {{
        background-color: transparent;
        color: inherit;
        padding: 0;
    }}

    img {{
        max-width: 95%;
        height: auto;
        display: block;
        margin: 12px auto 6px auto;
        border-radius: 6px;
        border: 1px solid #e2e8f0;
        page-break-inside: avoid;
    }}

    em {{
        color: #64748b;
        font-size: 12px;
        display: block;
        text-align: center;
        margin-bottom: 14px;
    }}

    hr {{
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 22px 0;
    }}

    ul, ol {{
        margin: 6px 0 10px 0;
        padding-left: 22px;
    }}

    li {{
        margin-bottom: 4px;
    }}

    strong {{
        color: #0f172a;
    }}

    a {{
        color: #2563eb;
        text-decoration: none;
    }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(full_html)

    if os.path.exists(EDGE_PATH):
        cmd = [
            EDGE_PATH,
            "--headless=new",
            "--disable-gpu",
            "--run-all-compositor-stages-before-draw",
            "--no-pdf-header-footer",
            f"--print-to-pdf={PDF_PATH}",
            HTML_PATH
        ]
        subprocess.run(cmd, capture_output=True, text=True)
        if os.path.exists(PDF_PATH):
            print(f"Generated {PDF_PATH}")

if __name__ == "__main__":
    main()
