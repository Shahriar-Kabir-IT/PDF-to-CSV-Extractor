import streamlit as st
import pdfplumber
import pandas as pd
from pathlib import Path
import re
from io import BytesIO
import requests
import base64
from PIL import Image
import fitz  # PyMuPDF
import os
import time
from datetime import datetime
import json
from llm_pdf_extractor import crop_pdf_to_images

# Page configuration
st.set_page_config(
    page_title="Commercial Invoice CSV Extractor",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("Commercial Invoice CSV Extractor")
st.markdown(
    """
    <style>
    /* Vibrant style for the download button only */
    [data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #00B4D8 0%, #0077B6 100%);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.1rem;
        font-weight: 600;
        letter-spacing: 0.2px;
        box-shadow: 0 8px 18px rgba(0, 119, 182, 0.35);
        transition: all 0.15s ease-in-out;
    }
    [data-testid="stDownloadButton"] > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 26px rgba(0, 119, 182, 0.50);
        filter: brightness(1.05);
    }

    /* Global polish */
    .stApp { background-color: #F6F9FC; }
    .section-title { color: #1F2937; font-weight: 600; letter-spacing: 0.2px; }
    .app-hr { border: none; height: 1px; background: #E5E7EB; margin: 12px 0; }

    /* Primary buttons (Proceed, Refresh) */
    [data-testid="stButton"] > button {
        background: linear-gradient(135deg, #00B4D8 0%, #0077B6 100%);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        letter-spacing: 0.2px;
        box-shadow: 0 8px 18px rgba(0, 119, 182, 0.25);
        transition: all 0.15s ease-in-out;
    }
    [data-testid="stButton"] > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 26px rgba(0, 119, 182, 0.45);
        filter: brightness(1.04);
    }

    /* File uploader container polish */
    [data-testid="stFileUploader"] > div {
        border: 2px dashed #CBD5E1;
        border-radius: 12px;
        padding: 14px;
        background: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Permanent refresh button for starting a new upload session
top_left, top_right = st.columns([3, 1])
with top_right:
    if st.button("🔄 Refresh to upload new document", use_container_width=True, key="refresh_top_button"):
        # Bump nonces to force widget reinitialization; clear auxiliary state
        st.session_state.uploader_nonce += 1
        st.session_state.filename_nonce += 1
        st.session_state['extracted_data'] = []
        st.session_state['last_saved_file'] = None
        # Trigger rerun
        try:
            st.experimental_rerun()
        except Exception:
            st.rerun()

# Initialize session state
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = []

if 'excel_save_path' not in st.session_state:
    st.session_state.excel_save_path = ""

if 'auto_save_enabled' not in st.session_state:
    st.session_state.auto_save_enabled = True

if 'last_save_time' not in st.session_state:
    st.session_state.last_save_time = None

if 'last_saved_file' not in st.session_state:
    st.session_state.last_saved_file = None

if 'overwrite_original' not in st.session_state:
    st.session_state.overwrite_original = False

# Widget key nonces to force reinitialization on refresh
if 'uploader_nonce' not in st.session_state:
    st.session_state.uploader_nonce = 0
if 'filename_nonce' not in st.session_state:
    st.session_state.filename_nonce = 0

def save_excel_automatically(df, original_filename=None, overwrite_original=False):
    """Automatically save the updated Excel file to a designated folder."""
    try:
        # Determine save path
        if st.session_state.excel_save_path:
            save_dir = Path(st.session_state.excel_save_path)
        else:
            # Default to a folder in the current directory
            save_dir = Path("updated_excel_files")
        
        # Create directory if it doesn't exist
        save_dir.mkdir(exist_ok=True)
        
        # Generate filename
        if original_filename and not overwrite_original:
            # Keep original name but add timestamp
            base_name = Path(original_filename).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{base_name}_updated_{timestamp}.xlsx"
        elif original_filename and overwrite_original:
            # Overwrite original file (use with caution)
            filename = original_filename
        else:
            filename = f"updated_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        file_path = save_dir / filename
        
        # Save the Excel file
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        
        # Update session state
        st.session_state.last_save_time = datetime.now()
        st.session_state.last_saved_file = str(file_path)
        
        return str(file_path), True
        
    except Exception as e:
        st.error(f"Error saving Excel file: {str(e)}")
        return None, False

def normalize_column_name(name):
    """Normalize column names by removing all underscores, spaces, and special chars, keeping only letters and numbers."""
    if not name:
        return ""
    # Remove all underscores, spaces, hyphens, and special characters
    normalized = re.sub(r'[_\s\-\.]+', '', str(name).lower())
    # Keep only alphanumeric characters
    normalized = re.sub(r'[^a-z0-9]', '', normalized)
    return normalized

def pdf_page_to_image(pdf_file, page_num=0):
    """Convert a PDF page to an image."""
    try:
        # Read PDF file
        pdf_bytes = pdf_file.read()
        pdf_file.seek(0)  # Reset file pointer
        
        # Open PDF with PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        if page_num >= len(doc):
            return None
        
        page = doc[page_num]
        
        # Convert to image (high resolution)
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(BytesIO(img_data))
        
        doc.close()
        return img
    except Exception as e:
        st.error(f"Error converting PDF to image: {str(e)}")
        return None

def extract_table_with_ai(image):
    """Use free AI API to extract table data from image."""
    try:
        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Use Hugging Face's free inference API with a table extraction model
        # Using a free model that can understand tables
        API_URL = "https://api-inference.huggingface.co/models/microsoft/table-transformer-structure-recognition"
        
        headers = {"Authorization": "Bearer hf_demo"}  # Free tier, no auth needed for some models
        
        # Try alternative: Use a simpler approach with OCR + table understanding
        # For now, let's use a different free API - Google's Vision API free tier
        # Or we can use a local OCR approach
        
        # Actually, let's use a simpler free API - Hugging Face's inference without auth
        # Or use Tesseract OCR which is free
        
        # For now, return None and we'll use the improved pdfplumber approach
        return None
        
    except Exception as e:
        return None

def extract_table_with_vision_api(pdf_file):
    """Extract table using vision-based approach - convert PDF to image and use OCR."""
    extracted_rows = []
    
    pdf_headers = [
        "Our Document",
        "Your Invoice No.",
        "Reference",
        "Cash Discount",
        "WHT amount",
        "Gross amount"
    ]
    
    excel_column_names = {
        "Our Document": "Our_Document",
        "Your Invoice No.": "Your_Invoice_No.",
        "Reference": "Reference",
        "Cash Discount": "Cash_Discount",
        "WHT amount": "WHT_amount",
        "Gross amount": "Gross_amount"
    }
    
    try:
        # Convert first page to image
        img = pdf_page_to_image(pdf_file, 0)
        if img is None:
            return []
        
        # For now, fall back to pdfplumber but with better logic
        # We'll improve the existing extraction instead
        return []
        
    except Exception as e:
        return []

def extract_table_from_pdf(pdf_file):
    """Extract table data from a PDF file using improved logic."""
    extracted_rows = []
    
    # PDF headers for data validation and cleaning
    pdf_headers = [
        "Our Document",
        "Your Invoice No.",
        "Reference",
        "Cash Discount",
        "WHT amount",
        "Gross amount"
    ]
    
    # Excel column name mapping
    excel_column_names = {
        "Our Document": "Our_Document",
        "Your Invoice No.": "Your_Invoice_No.",
        "Reference": "Reference",
        "Cash Discount": "Cash_Discount",
        "WHT amount": "WHT_amount",
        "Gross amount": "Gross_amount"
    }
    
    # Standard column names for invoice data
    standard_columns = [
        "Our Document",
        "Your Invoice No.",
        "Reference",
        "Cash Discount",
        "WHT amount",
        "Gross amount"
    ]
    
    # Column mapping for different variations
    column_variations = {
        "Our Document": ["Our Document", "Our_Document", "Document", "Doc No", "Document Number"],
        "Your Invoice No.": ["Your Invoice No.", "Your_Invoice_No.", "Invoice No", "Invoice Number", "Invoice"],
        "Reference": ["Reference", "Ref", "Reference No", "Ref No"],
        "Cash Discount": ["Cash Discount", "Cash_Discount", "Discount", "Disc"],
        "WHT amount": ["WHT amount", "WHT_amount", "WHT", "Tax", "Withholding"],
        "Gross amount": ["Gross amount", "Gross_amount", "Gross", "Amount", "Total"]
    }
    
    # Expected Excel column order
    expected_excel_columns = [
        'Our_Document', 'Your_Invoice_No.', 'Reference', 
        'Cash_Discount', 'WHT_amount', 'Gross_amount'
    ]

    # Numeric pattern supporting both US and EU formats
    number_pattern = r'(?:\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})|\d+(?:[.,]\d+)?)'

    def normalize_amount(val: str) -> str:
        """Normalize numeric strings to use '.' as decimal and remove thousand separators."""
        if not val:
            return ""
        s = str(val).strip()
        # If both '.' and ',' present, decide decimal by last separator
        if ',' in s and '.' in s:
            # Assume ',' is decimal if it appears after the last '.'
            if s.rfind(',') > s.rfind('.'):
                s = s.replace('.', '').replace(',', '.')
            else:
                s = s.replace(',', '')
        elif ',' in s and '.' not in s:
            # EU style: use ',' as decimal; remove thousand '.' if any
            s = s.replace('.', '').replace(',', '.')
        else:
            # US style or plain digits: remove commas
            s = s.replace(',', '')
        return s
    
    # Debug info storage
    debug_info = {
        'tables_found': 0,
        'total_rows_processed': 0,
        'rows_extracted': 0,
        'extraction_method': 'unknown',
        'error': None,
        'raw_tables': []  # Store raw table data for debugging
    }
    
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract tables from the page
                tables = page.extract_tables()
                
                debug_info['tables_found'] += len(tables)
                
                if tables:
                    for table_idx, table in enumerate(tables):
                        if not table or len(table) == 0:
                            continue
                        
                        debug_info['total_rows_processed'] += len(table)
                        
                        # Store raw table data for debugging
                        debug_info['raw_tables'].append({
                            'table_index': table_idx,
                            'rows': len(table),
                            'sample_data': table[:5] if len(table) > 0 else []
                        })
                        
                        # NEW APPROACH: Extract ALL data first, then try to map it
                        # Just extract everything as raw data rows
                        all_extracted_from_this_table = 0
                        
                        # Try each row as potential data (skip obvious header rows)
                        for row_idx, row in enumerate(table):
                            # Skip rows that are clearly headers (contain header keywords)
                            row_text = " ".join([str(cell).lower() for cell in row if cell])
                            if any(keyword in row_text for keyword in ['document', 'invoice', 'reference', 'discount', 'amount', 'sum', 'total']):
                                continue
                            # Skip generic headings commonly found in payment PDFs
                            if 'payment advice' in row_text:
                                continue
                            
                            # Create a data row from this table row
                            row_data = {}
                            
                            # Map by position - assume first columns are our target columns
                            for i, excel_col in enumerate(expected_excel_columns):
                                if i < len(row) and row[i] and str(row[i]).strip():
                                    value = str(row[i]).strip()
                                    # Clean the value
                                    if value.lower() not in [h.lower() for h in pdf_headers]:
                                        row_data[excel_col] = value
                            
                            # Also try to extract any numeric values that look like amounts
                            for i, cell in enumerate(row):
                                if cell and str(cell).strip():
                                    cell_value = str(cell).strip()
                                    # Look for numeric patterns
                                    if re.search(r'\d+\.?\d*', cell_value):
                                        # Try to assign to appropriate column based on position
                                        if i == len(row) - 1:  # Last column - likely gross amount
                                            row_data['Gross_amount'] = cell_value
                                        elif i == len(row) - 2:  # Second to last - likely WHT
                                            row_data['WHT_amount'] = cell_value
                                        elif i == len(row) - 3:  # Third to last - likely discount
                                            row_data['Cash_Discount'] = cell_value
                                    
                                    # Look for document numbers (must include digits and be code-like)
                                    if re.search(r'(?=.*\d)[A-Za-z0-9-]{4,}', cell_value) and ' ' not in cell_value:
                                        if 'Our_Document' not in row_data:
                                            row_data['Our_Document'] = cell_value
                                    
                                    # Look for invoice numbers
                                    if re.search(r'\d+', cell_value) and len(cell_value) > 3:
                                        if 'Your_Invoice_No.' not in row_data:
                                            row_data['Your_Invoice_No.'] = cell_value
                            
                            # Only keep rows that have actual data (not just headers or junk)
                            if row_data and any(row_data.values()):
                                # Require at least one amount or two key identifiers before accepting the row
                                valid_fields = sum(1 for k in ['Our_Document','Your_Invoice_No.','Gross_amount','WHT_amount','Cash_Discount'] if row_data.get(k))
                                if valid_fields >= 2 or row_data.get('Gross_amount'):
                                    # Fill in missing columns with empty strings
                                    for col in expected_excel_columns:
                                        if col not in row_data:
                                            row_data[col] = ""
                                    extracted_rows.append(row_data)
                                    all_extracted_from_this_table += 1
                        
                        debug_info['rows_extracted'] += all_extracted_from_this_table
                        
                        # If we extracted data from this table, don't try other tables
                        if all_extracted_from_this_table > 0:
                            debug_info['extraction_method'] = f'raw_extraction_table_{table_idx}'
                            break
        
        # Also try text extraction as fallback for tables that aren't detected
        if not extracted_rows:
            debug_info['extraction_method'] = 'text_fallback'
            with pdfplumber.open(pdf_file) as pdf:
                fallback_rows = 0
                for page_num, page in enumerate(pdf.pages, 1):
                    # Positional extraction using word coordinates
                    words = page.extract_words(use_text_flow=True)
                    if words:
                        # Group words by approximate line using y0
                        lines = {}
                        for w in words:
                            y = int(round(w.get('top', 0)))
                            lines.setdefault(y, []).append(w)
                        sorted_lines = sorted(lines.items(), key=lambda x: x[0])

                        # Find header line: contains at least 3 header keywords
                        header_idx = None
                        header_x_positions = []
                        for idx, (y, wlist) in enumerate(sorted_lines):
                            line_text = ' '.join([w['text'] for w in wlist])
                            hits = sum(1 for h in [
                                'document','invoice','reference','discount','wht','gross'
                            ] if h in line_text.lower())
                            if hits >= 3:
                                header_idx = idx
                                # Record representative x positions for expected headers using keyword mapping
                                keyword_order = ['document','invoice','reference','discount','wht','gross']
                                keyword_pos = {k: None for k in keyword_order}
                                for w in wlist:
                                    t = w['text'].lower()
                                    for k in keyword_order:
                                        if k in t:
                                            if keyword_pos[k] is None:
                                                keyword_pos[k] = w['x0']
                                            else:
                                                keyword_pos[k] = min(keyword_pos[k], w['x0'])
                                header_x_positions = [keyword_pos[k] for k in keyword_order if keyword_pos[k] is not None]
                                break

                        # If header found, build column boundaries and extract subsequent lines
                        if header_idx is not None:
                            # Map boundaries to excel columns in expected order
                            boundary_cols = [
                                'Our_Document','Your_Invoice_No.','Reference',
                                'Cash_Discount','WHT_amount','Gross_amount'
                            ]
                            # Build six positions: prefer detected header_x_positions, else evenly spaced
                            if header_x_positions and len(header_x_positions) >= 2:
                                positions = sorted(header_x_positions)
                                if len(positions) > len(boundary_cols):
                                    positions = positions[:len(boundary_cols)]
                                if len(positions) < len(boundary_cols):
                                    xmin = min(positions)
                                    xmax = max(positions)
                                    step = (xmax - xmin) / (len(boundary_cols) - 1) if len(boundary_cols) > 1 else (xmax - xmin)
                                    positions = [xmin + i * step for i in range(len(boundary_cols))]
                            else:
                                y_hdr, wlist_hdr = sorted_lines[header_idx]
                                if wlist_hdr:
                                    xmin = min(w['x0'] for w in wlist_hdr)
                                    xmax = max(w.get('x1', w['x0'] + 20) for w in wlist_hdr)
                                else:
                                    xmin, xmax = 0, 600
                                step = (xmax - xmin) / (len(boundary_cols) - 1) if len(boundary_cols) > 1 else (xmax - xmin)
                                positions = [xmin + i * step for i in range(len(boundary_cols))]

                            boundaries = []
                            positions = sorted(positions)
                            for i, x in enumerate(positions):
                                if i < len(positions) - 1:
                                    mid = (x + positions[i+1]) / 2.0
                                else:
                                    mid = x + 1000
                                boundaries.append((x, mid))
                            # Process lines after header
                            for idx in range(header_idx+1, len(sorted_lines)):
                                y, wlist = sorted_lines[idx]
                                line_text = ' '.join([w['text'] for w in wlist])
                                # Skip lines without digits
                                if not re.search(r'\d', line_text):
                                    continue
                                # Build segments per boundary
                                segments = ['' for _ in boundary_cols]
                                for w in wlist:
                                    x = w['x0']
                                    # find boundary index
                                    bidx = None
                                    for j, (x0, xmid) in enumerate(boundaries):
                                        if x >= x0 and x < xmid:
                                            bidx = j
                                            break
                                    if bidx is None:
                                        continue
                                    if 0 <= bidx < len(segments):
                                        segments[bidx] = (segments[bidx] + ' ' + w['text']).strip()

                                row_data = {}
                                # Assign and normalize amounts
                                for cidx, col in enumerate(boundary_cols):
                                    val = segments[cidx].strip()
                                    if col in ['Gross_amount','WHT_amount','Cash_Discount']:
                                        m = re.search(number_pattern, val)
                                        if m:
                                            row_data[col] = normalize_amount(m.group(0))
                                        else:
                                            row_data[col] = ''
                                    elif col == 'Our_Document':
                                        # Prefer alpha-numeric code without spaces
                                        m = re.search(r'(?=.*\d)[A-Za-z0-9\-]{4,}', val)
                                        row_data[col] = m.group(0) if m else val.strip()
                                    elif col == 'Your_Invoice_No.':
                                        m = re.search(r'\d{4,}', val)
                                        row_data[col] = m.group(0) if m else val.strip()
                                    else:
                                        row_data[col] = val.strip()

                                # Validate row: must have Invoice No. and Gross amount
                                if row_data.get('Your_Invoice_No.') and row_data.get('Gross_amount'):
                                    # Fill missing columns
                                    for col in expected_excel_columns:
                                        if col not in row_data:
                                            row_data[col] = ''
                                    extracted_rows.append(row_data)
                                    fallback_rows += 1
                        else:
                            # Fallback simple line-based extraction
                            text = page.extract_text()
                            if text:
                                for line in text.split('\n'):
                                    line_lower = line.lower()
                                    if 'payment advice' in line_lower:
                                        continue
                                    if not re.search(r'\d', line_lower):
                                        continue
                                    row_data = {}
                                    # Extract amounts by capturing last three numeric tokens
                                    nums = re.findall(number_pattern, line)
                                    if nums:
                                        # Assign from right to left: gross, wht, discount where applicable
                                        nums_norm = [normalize_amount(n) for n in nums]
                                        if len(nums_norm) >= 1:
                                            row_data['Gross_amount'] = nums_norm[-1]
                                        if len(nums_norm) >= 2:
                                            row_data['WHT_amount'] = nums_norm[-2]
                                        if len(nums_norm) >= 3:
                                            row_data['Cash_Discount'] = nums_norm[-3]
                                    # Document and invoice patterns
                                    mdoc = re.search(r'(?=.*\d)[A-Za-z0-9\-]{4,}', line)
                                    if mdoc:
                                        row_data['Our_Document'] = mdoc.group(0)
                                    # Invoice number: prefer plain integers (4-8 digits) without separators
                                    invoice_candidate = None
                                    for tok in re.findall(r'\b[0-9][0-9.,]*\b', line):
                                        if ('.' in tok) or (',' in tok):
                                            continue
                                        if 4 <= len(tok) <= 8:
                                            invoice_candidate = tok
                                            break
                                    if invoice_candidate:
                                        row_data['Your_Invoice_No.'] = invoice_candidate
                                    # Reference loosely as any non-amount alphanumeric fragment
                                    if 'Reference' not in row_data:
                                        mref = re.search(r'Reference[:\s]+([^\n]+)', line, re.IGNORECASE)
                                        if mref:
                                            row_data['Reference'] = mref.group(1).strip()
                                    # Validate row
                                    if row_data and any(row_data.values()):
                                        # Require Invoice No. and Gross amount to avoid totals
                                        if row_data.get('Your_Invoice_No.') and row_data.get('Gross_amount'):
                                            for col in expected_excel_columns:
                                                if col not in row_data:
                                                    row_data[col] = ''
                                            extracted_rows.append(row_data)
                                            fallback_rows += 1
                
                debug_info['rows_extracted'] += fallback_rows

    except Exception as e:
        debug_info['error'] = str(e)
        st.error(f"Error processing {pdf_file.name}: {str(e)}")
        return [], debug_info

    # If still nothing extracted, try local Ollama (if running) to parse text
    if not extracted_rows:
        try:
            with pdfplumber.open(pdf_file) as pdf:
                full_text = []
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        full_text.append(t)
                full_text = '\n'.join(full_text)
            rows = []
            if full_text:
                # Ask Ollama to return JSON rows
                prompt = (
                    "You are a parser. Extract invoice payment rows from the following text. "
                    "Return a JSON array where each object has keys: Our_Document, Your_Invoice_No., "
                    "Reference, Cash_Discount, WHT_amount, Gross_amount. Use amounts as strings and "
                    "keep original codes. Only include plausible data rows; skip headers.\n\nTEXT:\n" + full_text
                )
                resp = requests.post(
                    'http://localhost:11434/api/generate',
                    json={
                        'model': 'llama3',
                        'prompt': prompt,
                        'stream': False
                    },
                    timeout=5
                )
                if resp.status_code == 200:
                    data = resp.json()
                    out = data.get('response', '').strip()
                    # Attempt to locate JSON array in response
                    json_start = out.find('[')
                    json_end = out.rfind(']')
                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        try:
                            rows = json.loads(out[json_start:json_end+1])
                        except Exception:
                            rows = []
            # Normalize and accept rows
            for r in rows:
                row_data = {}
                for col in expected_excel_columns:
                    val = r.get(col, r.get(col.replace('_',' '), ''))
                    if col in ['Gross_amount','WHT_amount','Cash_Discount']:
                        row_data[col] = normalize_amount(val)
                    else:
                        row_data[col] = str(val).strip() if val is not None else ''
                if any(row_data.values()):
                    extracted_rows.append(row_data)
        except Exception:
            # Ignore Ollama errors silently
            pass

    return extracted_rows, debug_info

def extract_table_via_api(
    pdf_file,
    api_url: str = "http://127.0.0.1:8000/extract",
    bbox: str | None = None,
    scale: float = 3.0,
    use_ollama: bool = False,
    api_key: str | None = None,
):
    """Call the local FastAPI extractor and return (rows, debug_info)."""
    debug_info = {
        'method': 'api',
        'api_url': api_url,
        'rows_extracted': 0,
        'error': None,
        'csv_path': None,
    }
    try:
        # Streamlit UploadedFile supports getvalue() to read bytes safely
        pdf_bytes = pdf_file.getvalue()
        files = {
            'file': (pdf_file.name, pdf_bytes, 'application/pdf')
        }
        data = {
            'save_csv_flag': True,
            'scale': scale,
            'use_ollama': use_ollama,
        }
        if bbox:
            data['bbox'] = bbox
        if api_key:
            data['api_key'] = api_key
        resp = requests.post(api_url, files=files, data=data, timeout=60)
        resp.raise_for_status()
        js = resp.json()
        rows = js.get('rows', [])
        debug_info['rows_extracted'] = len(rows)
        debug_info['csv_path'] = js.get('csv_path')
        return rows, debug_info
    except Exception as e:
        debug_info['error'] = str(e)
        st.error(f"API error for {pdf_file.name}: {e}")
        return [], debug_info

def match_and_update_excel(excel_df, extracted_rows, debug_mode=False):
    """
    Match extracted data to Excel columns and update the DataFrame.
    Handles both existing columns and creates new ones if needed.
    """
    # Standard column names for invoice data
    standard_columns = [
        "Our Document",
        "Your Invoice No.",
        "Reference",
        "Cash Discount",
        "WHT amount",
        "Gross amount"
    ]
    
    # Column mapping for different variations
    column_variations = {
        "Our Document": ["Our Document", "Our_Document", "Document", "Doc No", "Document Number"],
        "Your Invoice No.": ["Your Invoice No.", "Your_Invoice_No.", "Invoice No", "Invoice Number", "Invoice"],
        "Reference": ["Reference", "Ref", "Reference No", "Ref No"],
        "Cash Discount": ["Cash Discount", "Cash_Discount", "Discount", "Disc"],
        "WHT amount": ["WHT amount", "WHT_amount", "WHT", "Tax", "Withholding"],
        "Gross amount": ["Gross amount", "Gross_amount", "Gross", "Amount", "Total"]
    }
    
    if debug_mode:
        st.write("### 📊 Data Matching Process")
        st.write(f"Extracted rows: {len(extracted_rows)}")
        st.write(f"Excel columns: {list(excel_df.columns)}")
    
    # Create a copy to avoid modifying the original
    updated_df = excel_df.copy()
    
    # Track new columns added
    new_columns = []
    
    # Process each extracted row
    for row_idx, extracted_row in enumerate(extracted_rows):
        if debug_mode:
            st.write(f"**Row {row_idx + 1}:** {extracted_row}")
        
        # Create a new row for the Excel DataFrame
        new_row = {}
        
        def get_row_value(row, standard_col):
            candidates = [
                standard_col,
                standard_col.replace(' ', '_'),
                standard_col.replace(' ', '_').replace('.', '')
            ]
            for var in column_variations.get(standard_col, []):
                candidates.extend([
                    var,
                    var.replace(' ', '_'),
                    var.replace(' ', '_').replace('.', '')
                ])
            for key in candidates:
                if key in row and row.get(key) not in [None, ""]:
                    return row.get(key)
            return ""

        # Map extracted data to Excel columns
        for standard_col in standard_columns:
            # Find matching column in existing Excel (case-insensitive, space-insensitive)
            excel_col = None
            for existing_col in excel_df.columns:
                # Check for exact match or variations
                if (existing_col.lower().replace(' ', '_') == standard_col.lower().replace(' ', '_') or
                    existing_col.lower() == standard_col.lower() or
                    existing_col in column_variations.get(standard_col, [])):
                    excel_col = existing_col
                    break
            
            # If column doesn't exist, create it with standard name
            if excel_col is None:
                excel_col = standard_col.replace(' ', '_').replace('.', '')
                if excel_col not in updated_df.columns:
                    updated_df[excel_col] = ""
                    new_columns.append(excel_col)
            
            # Get the value from extracted row
            value = get_row_value(extracted_row, standard_col)
            new_row[excel_col] = value
        
        # Add the row to DataFrame
        if new_row:
            updated_df = pd.concat([updated_df, pd.DataFrame([new_row])], ignore_index=True)
    
    if debug_mode:
        if new_columns:
            st.success(f"✅ Added new columns: {new_columns}")
        st.write(f"📊 Total rows in updated Excel: {len(updated_df)}")
    
    return updated_df

# Clean header only; instructional text removed for professional UI

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Upload PDF Files")
    pdf_files = st.file_uploader(
        "Choose PDF files with invoice data",
        type=['pdf'],
        accept_multiple_files=True,
        key=f"pdf_files_widget_{st.session_state.uploader_nonce}",
        help="Select one or more PDF files containing invoice information"
    )

with col2:
    st.subheader("📁 Output")
    # Keep backend defaults internal; remove from UI
    use_api = True
    api_url = "http://127.0.0.1:8000/extract"
    bbox_str = ""
    scale_val = 3.0
    use_ollama = False

    # Always CSV-only; require an explicit filename from the user
    csv_only = True
    csv_filename_input = st.text_input(
        "CSV filename (without .csv)",
        value="",
        placeholder="e.g., Payments_Oct_2025",
        key=f"csv_filename_input_{st.session_state.filename_nonce}",
        help="Enter the CSV name to use for the download"
    ).strip()

    # Hide Excel upload and autosave configuration when CSV-only is enabled
    if not csv_only:
        st.divider()
        st.subheader("📊 Select Excel File")
        excel_option = st.radio(
            "Choose Excel option:",
            ["Upload existing Excel file", "Create new Excel file"],
            help="Use existing template or create new Excel file"
        )
        
        if excel_option == "Upload existing Excel file":
            excel_file = st.file_uploader(
                "Choose Excel template",
                type=['xlsx', 'xls'],
                accept_multiple_files=False,
                help="Select Excel file with your column headers"
            )
        else:
            excel_file = None
            st.info("✅ A new Excel file will be created with standard columns")
        
        st.divider()
        
        st.header("⚙️ Auto-Save Settings (Excel)")
        
        # Auto-save toggle
        st.session_state.auto_save_enabled = st.checkbox(
            "Enable Auto-Save",
            value=st.session_state.auto_save_enabled,
            help="Automatically save updated Excel files without requiring download"
        )
        
        # Overwrite original file option
        if st.session_state.auto_save_enabled:
            st.session_state.overwrite_original = st.checkbox(
                "Overwrite Original Excel File",
                value=False,
                help="⚠️ Warning: This will replace the original Excel file with the updated version"
            )
        
        # Excel save path
        save_path_input = st.text_input(
            "Excel Save Folder Path (optional)",
            value=st.session_state.excel_save_path,
            placeholder="Leave empty to use default 'updated_excel_files' folder",
            help="Enter full folder path where updated Excel files will be saved"
        )
        
        if save_path_input != st.session_state.excel_save_path:
            st.session_state.excel_save_path = save_path_input.strip()
        
        # Show current save location
        if st.session_state.auto_save_enabled:
            if st.session_state.excel_save_path:
                st.info(f"📁 Files will be saved to: {st.session_state.excel_save_path}")
            else:
                st.info("📁 Files will be saved to: ./updated_excel_files/")
            
            if st.session_state.overwrite_original:
                st.warning("⚠️ Original file overwrite mode enabled")
        
        st.divider()

    # Preview removed for a cleaner, professional UI

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.write("")

with col2:
    # Excel preview removed entirely
    excel_df = pd.DataFrame()

# Process button
process_button = st.button("🚀 Proceed", type="primary", use_container_width=True)

if process_button:
    if not pdf_files:
        st.error("❌ Please upload at least one PDF file.")
    elif not csv_filename_input:
        st.error("❌ Please provide a CSV filename before proceeding.")
    else:
        with st.spinner("🔄 Processing your files..."):
            # Helper for natural (human) filename sort: 1,2,10 vs 1,10,2
            def _natural_sort_key(name: str):
                parts = re.findall(r"\d+|\D+", name)
                return [int(p) if p.isdigit() else p.lower() for p in parts]
            # Initialize Excel DataFrame only if not CSV-only
            if not csv_only:
                if excel_option == "Upload existing Excel file" and excel_file:
                    try:
                        excel_df = pd.read_excel(excel_file)
                        st.success(f"✅ Loaded Excel file with {len(excel_df)} existing rows")
                    except Exception as e:
                        st.error(f"❌ Error loading Excel file: {str(e)}")
                        excel_df = pd.DataFrame()
                else:
                    # Create new DataFrame with standard columns
                    excel_df = pd.DataFrame(columns=[col.replace(' ', '_').replace('.', '') for col in standard_columns])
                    st.info("ℹ️ Creating new Excel file with standard columns")
            
            # Process each PDF
            total_extracted = 0
            progress_bar = st.progress(0)
            # Minimal status; no verbose messages
            merged_rows = [] if csv_only else None
            merged_filename_base = csv_filename_input
            
            # Maintain serial order by sorting by filename naturally (ascending)
            ordered_pdf_files = sorted(pdf_files, key=lambda f: _natural_sort_key(getattr(f, "name", str(f))))

            for idx, pdf_file in enumerate(ordered_pdf_files):
                
                # Extract data from PDF
                if use_api:
                    extracted_rows, debug_info = extract_table_via_api(
                        pdf_file,
                        api_url=api_url,
                        bbox=bbox_str.strip() or None,
                        scale=scale_val,
                        use_ollama=use_ollama,
                        api_key=os.getenv('OPENAI_API_KEY')
                    )
                    # Graceful fallback if API unreachable or returned zero rows
                    if debug_info.get('error') or not extracted_rows:
                        st.info("API unavailable or returned no rows; falling back to local extractor.")
                        extracted_rows, debug_info = extract_table_from_pdf(pdf_file)
                else:
                    extracted_rows, debug_info = extract_table_from_pdf(pdf_file)

                # CSV-only output: accumulate rows for a single merged CSV
                if csv_only:
                    if extracted_rows:
                        total_extracted += len(extracted_rows)
                        merged_rows.extend(extracted_rows)
                else:
                    # Update Excel DataFrame
                    if extracted_rows:
                        excel_df = match_and_update_excel(excel_df, extracted_rows, debug_mode=False)
                        total_extracted += len(extracted_rows)
                    
                
                # Update progress
                progress_bar.progress((idx + 1) / len(ordered_pdf_files))
            
            # Final results
            st.divider()
            
            if total_extracted > 0:
                if csv_only:
                    # Build merged DataFrame and provide professional download button
                    merged_df = pd.DataFrame(merged_rows)
                    out_dir = os.path.join(os.getcwd(), 'updated_excel_files')
                    os.makedirs(out_dir, exist_ok=True)
                    merged_filename = f"{merged_filename_base}.csv"
                    merged_path = os.path.join(out_dir, merged_filename)
                    merged_df.to_csv(merged_path, index=False)
                    csv_bytes = merged_df.to_csv(index=False).encode('utf-8')

                    dl_col, refresh_col = st.columns([3, 1])
                    with dl_col:
                        st.download_button(
                            label="Download CSV",
                            data=csv_bytes,
                            file_name=merged_filename,
                            mime='text/csv',
                            use_container_width=True
                        )
                    with refresh_col:
                        if st.button("🔄 Refresh to upload new document", use_container_width=True, key="refresh_bottom_button"):
                            # Bump nonces to force widget reinitialization; clear auxiliary state
                            st.session_state.uploader_nonce += 1
                            st.session_state.filename_nonce += 1
                            st.session_state['extracted_data'] = []
                            st.session_state['last_saved_file'] = None
                            # Trigger rerun
                            try:
                                st.experimental_rerun()
                            except Exception:
                                # Fallback for newer Streamlit versions
                                st.rerun()
                else:
                    # Show data preview for Excel flow
                    if not excel_df.empty:
                        st.write("### 📊 Data Preview")
                        st.dataframe(excel_df.head(10))

                        # Download section
                        st.write("### 💾 Download Results")
                        col1, col2 = st.columns(2)

                        with col1:
                            # Download as Excel
                            output_buffer = BytesIO()
                            excel_df.to_excel(output_buffer, index=False, engine='openpyxl')
                            output_buffer.seek(0)

                            st.download_button(
                                label="📥 Download Excel File",
                                data=output_buffer,
                                file_name=f"invoice_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )

                        with col2:
                            # Download as CSV
                            csv_buffer = BytesIO()
                            excel_df.to_csv(csv_buffer, index=False)
                            csv_buffer.seek(0)

                            st.download_button(
                                label="📄 Download CSV File",
                                data=csv_buffer,
                                file_name=f"invoice_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

                        # Auto-save using configured save settings
                        if st.session_state.auto_save_enabled:
                            saved_path, ok = save_excel_automatically(
                                excel_df,
                                original_filename=(excel_file.name if excel_file else None),
                                overwrite_original=st.session_state.overwrite_original
                            )
                            if ok and saved_path:
                                st.info(f"💾 File saved to: `{saved_path}`")
                            else:
                                st.warning("⚠️ Auto-save failed. Use the download buttons above.")
                        else:
                            st.info("💾 Auto-save is disabled. Use the download buttons above.")
                        
            else:
                st.error("❌ No data was extracted. Please check your PDF files and try again.")
                st.info("💡 Tips: Make sure your PDFs contain invoice data in table format")

# Display session state data if available
if st.session_state.extracted_data:
    with st.expander("📋 View All Extracted Data"):
        st.dataframe(pd.DataFrame(st.session_state.extracted_data), use_container_width=True)

# Show recent saved files and file info
if st.session_state.auto_save_enabled and st.session_state.last_saved_file:
    with st.expander("📁 Recent Saved Files", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📝 Last saved file: {Path(st.session_state.last_saved_file).name}")
            st.caption(f"📍 Location: {st.session_state.last_saved_file}")
        with col2:
            if st.session_state.last_save_time:
                st.info(f"⏰ Last saved: {st.session_state.last_save_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Show file size and allow opening folder
        if Path(st.session_state.last_saved_file).exists():
            file_size = Path(st.session_state.last_saved_file).stat().st_size / 1024
            st.caption(f"📊 File size: {file_size:.1f} KB")
            
            # Button to open folder (works on most systems)
            folder_path = str(Path(st.session_state.last_saved_file).parent)
            if st.button("📂 Open Folder Location", use_container_width=True):
                try:
                    import subprocess
                    import platform
                    if platform.system() == 'Windows':
                        subprocess.Popen(['explorer', folder_path])
                    elif platform.system() == 'Darwin':  # macOS
                        subprocess.Popen(['open', folder_path])
                    else:  # Linux
                        subprocess.Popen(['xdg-open', folder_path])
                    st.success(f"📂 Opening folder: {folder_path}")
                except Exception as e:
                    st.error(f"Could not open folder: {str(e)}")
                    st.info(f"📍 Folder path: {folder_path}")

# Removed listing of all saved Excel files for a cleaner, CSV-only UI

