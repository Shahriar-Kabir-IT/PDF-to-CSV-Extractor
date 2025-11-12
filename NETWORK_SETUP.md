# PDF to Excel Extractor - Network Setup Guide

## 🌐 Network Access Instructions

Your PDF to Excel Extractor is now running in **network mode** and can be accessed by other users on your local network.

### 🔗 Access URLs

- **Local Access**: http://localhost:8505
- **Network Access**: http://192.168.10.119:8505

### 📋 How Other Users Can Connect

1. **Same Network**: Ensure all users are connected to the same local network
2. **Firewall**: Make sure your firewall allows connections on port 8505
3. **Browser Access**: Other users can access the application by typing your computer's IP address in their browser:
   ```
   http://YOUR_COMPUTER_IP:8505
   ```

### 🚀 Quick Start for Users

#### For Windows Users:
Simply double-click `start_network_server.bat` to start the server in network mode.

#### For Manual Start:
```bash
python network_config.py
```

### 📊 Simple 3-Step Process

1. **Upload PDF Files**: Users select one or more PDF files containing invoice data
2. **Choose Excel Option**: 
   - Upload existing Excel template, OR
   - Create new Excel file with standard columns
3. **Process & Download**: Click process and download the extracted data

### 📋 Standard Columns

The application automatically handles these columns:
- **Our Document** - Document numbers
- **Your Invoice No.** - Invoice numbers  
- **Reference** - Reference numbers
- **Cash Discount** - Discount amounts
- **WHT amount** - Withholding tax amounts
- **Gross amount** - Total amounts

### 🔧 Features

✅ **Automatic Column Detection** - Finds existing columns in Excel files  
✅ **Smart Column Creation** - Adds missing columns automatically  
✅ **Multiple PDF Support** - Process many PDFs at once  
✅ **Data Preview** - Shows extracted data before download  
✅ **Multiple Formats** - Download as Excel (.xlsx) or CSV (.csv)  
✅ **Auto-Save** - Automatically saves files locally  

### 🛠️ Troubleshooting

#### If Users Can't Connect:
1. Check that you're all on the same network
2. Verify firewall settings allow port 8505
3. Try accessing with `localhost:8505` first
4. Check your computer's IP address with `ipconfig` (Windows) or `ifconfig` (Mac/Linux)

#### If No Data Extracts:
1. Ensure PDFs contain table-formatted data
2. Check that PDFs aren't scanned images (use OCR if needed)
3. Try the debug mode to see what tables are detected

### 🔒 Security Notes

- This is designed for **local network use only**
- The server binds to all network interfaces (0.0.0.0)
- No authentication is implemented - use only on trusted networks
- For production use, consider adding authentication and HTTPS

### 📁 File Locations

- **Auto-saved files**: Saved in the application folder with timestamp
- **Default naming**: `invoice_extracted_YYYYMMDD_HHMMSS.xlsx`
- **Configuration**: Edit `network_config.py` to change ports or settings

---

**Need Help?** Check the main README.md for detailed troubleshooting or contact your system administrator.