# Quick Start Guide - Network Deployment

## 🚀 Quick Setup (Windows)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Find your IP address:**
   - Press `Win + R`, type `cmd`, press Enter
   - Type `ipconfig` and press Enter
   - Look for "IPv4 Address" (e.g., 192.168.1.100)

3. **Start the server:**
   - Double-click `start_server.bat`
   - Or run: `python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501`

4. **Share the URL with users:**
   - Tell them to open: `http://[YOUR_IP]:8501`
   - Example: `http://192.168.1.100:8501`

## 🔥 Firewall Setup (Important!)

If users can't connect, allow Python through Windows Firewall:

1. Open **Windows Defender Firewall**
2. Click **Allow an app or feature through Windows Defender Firewall**
3. Click **Change settings** → **Allow another app**
4. Browse to your Python executable (usually in `C:\Users\[YourName]\AppData\Local\Programs\Python\`)
5. Add both `python.exe` and `pythonw.exe`
6. Check both **Private** and **Public** networks
7. Click **OK**

Or allow port 8501:
1. Open **Windows Defender Firewall** → **Advanced settings**
2. Click **Inbound Rules** → **New Rule**
3. Select **Port** → **TCP** → **Specific local ports: 8501**
4. Allow the connection → Apply to all profiles → Name it "Streamlit"

## ✅ Verify It's Working

1. On your PC: Open `http://localhost:8501` - should work
2. On another PC: Open `http://[YOUR_IP]:8501` - should work
3. If step 2 fails, check firewall settings above

## 📝 Notes

- Keep the server running on your PC
- All users can access simultaneously
- Each user has their own session
- Files are processed on the server PC

