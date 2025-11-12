# PDF Table Data Extractor

A web-based Python application that extracts specific tabular data from PDF documents and populates Excel files. This application uses **no paid APIs** - it relies entirely on open-source libraries for PDF processing.

## Features

- 📄 Upload multiple PDF files
- 📋 Upload an Excel file template (.xlsx)
- 🔍 Automatically extract data matching specific columns:
  - Our Document
  - Your Invoice No.
  - Reference
  - Cash Discount
  - WHT amount
  - Gross amount
- 💾 Download updated Excel file with extracted data
- 🌐 Simple web interface built with Streamlit

## Installation

1. Install Python 3.8 or higher

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### For Local Use (Single User)

1. Start the Streamlit application:
```bash
python -m streamlit run app.py
```

2. The application will open in your web browser automatically

### For Network Deployment (Multiple Users)

**On the Server PC (Your PC):**

1. **Find your PC's IP address:**
   - **Windows**: Open Command Prompt and run `ipconfig`. Look for "IPv4 Address" under your active network adapter
   - **Linux/Mac**: Run `hostname -I` or `ifconfig` in terminal

2. **Start the server:**
   - **Windows**: Double-click `start_server.bat` or run:
     ```bash
     python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
     ```
   - **Linux/Mac**: Run:
     ```bash
     chmod +x start_server.sh
     ./start_server.sh
     ```
     Or manually:
     ```bash
     python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
     ```

3. **Note the IP address and port** shown in the terminal (e.g., `http://192.168.1.100:8501`)

**On Client PCs (Other Users on the Network):**

1. Open a web browser
2. Navigate to: `http://[SERVER_IP_ADDRESS]:8501`
   - Replace `[SERVER_IP_ADDRESS]` with the IP address from step 3 above
   - Example: `http://192.168.1.100:8501`

**Important Notes:**
- The server PC must keep the application running
- All users must be on the same network
- Windows Firewall may block the connection - you may need to allow Python/Streamlit through the firewall
- Each user will have their own session and can upload files independently

### Using the Web Interface

1. **Select PDF files**: Click "Select PDF files" and choose one or more PDF files containing the tables you want to extract
2. **Select Excel file**: Click "Select Excel file" and choose your Excel template file (.xlsx format)
3. **Click "Process Files"**: The application will extract data from PDFs and match it to your Excel columns
4. **Download Updated Excel**: After processing, click "Download Updated Excel" to save the results

## How It Works

1. **PDF Processing**: Uses `pdfplumber` to detect and extract tables from PDF pages
2. **Column Matching**: Intelligently matches extracted columns to your Excel columns (handles variations in spacing, underscores, etc.)
3. **Data Extraction**: Extracts rows containing data under the specified column headers
4. **Excel Update**: Appends extracted data to your Excel file, matching columns automatically

## Requirements

- Python 3.8+
- streamlit
- pdfplumber
- pandas
- pymupdf (optional, for additional PDF support)
- openpyxl (for Excel file support)

## Notes

- The application looks for tables in PDFs and matches column names flexibly (ignoring spaces, underscores, and case differences)
- If a table isn't automatically detected, the application will try text-based extraction as a fallback
- Each extracted row includes metadata: Source PDF name and page number

## Troubleshooting

### Application Issues

- **No data extracted**: Make sure your PDFs contain tables with the expected column headers
- **Column mismatch**: The application tries to match columns flexibly, but ensure your Excel file has similar column names
- **Processing errors**: Check that your PDFs are not corrupted or password-protected
- **Excel read errors**: Make sure your Excel file is not open in another program and is in .xlsx format (convert .xls files to .xlsx if needed)

### Network Access Issues

- **Cannot access from other PCs**: 
  - Check that Windows Firewall allows Python/Streamlit (add exception for port 8501)
  - Verify all PCs are on the same network
  - Try accessing using the server PC's IP address instead of localhost
  - Check that the server is running and shows "0.0.0.0" as the address

- **Connection refused**:
  - Make sure the server PC is still running the application
  - Verify the IP address and port number are correct
  - Check if another application is using port 8501 (change port in config.toml if needed)

- **Firewall blocking**:
  - **Windows**: Go to Windows Defender Firewall → Allow an app → Add Python/Streamlit
  - Or allow port 8501 through firewall settings

