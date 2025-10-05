# 🌤️ Project-BlueSky-Minds - Installation & Usage Guide

A revolutionary weather probability assessment tool using 30+ years of NASA data to predict weather patterns months in advance.

---

## 📋 Table of Contents

- [What is Project-BlueSky-Minds?](#-what-is-weather-odds-pro)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the Application](#️-running-the-application)
- [Using the Application](#-using-the-application)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 What is Project-BlueSky-Minds?

Project-BlueSky-Minds is **NOT** another weather forecast app. It's a probability-based risk assessment tool that:

- ✅ Analyzes **30+ years** of NASA historical weather data
- ✅ Predicts weather patterns **2-3 months** in advance
- ✅ Provides **statistical probabilities** rather than deterministic forecasts
- ✅ Perfect for event planning, agriculture, and climate analysis

**Example Use Cases:**
- Planning outdoor weddings months in advance
- Agricultural planting/harvesting decisions
- Construction project scheduling
- Climate change trend analysis

---

## 📦 Prerequisites

Make sure you have these installed before starting:

| Software | Version | Check Command | Download Link |
|----------|---------|---------------|---------------|
| **Python** | 3.11.9 or higher | `python --version` | [Download Python](https://www.python.org/downloads/) |
| **pip** | Latest | `pip --version` | (Included with Python) |
| **Git** | Any version | `git --version` | [Download Git](https://git-scm.com/downloads) |
| **Redis** | 6.0 or higher | `redis-server --version` | [Download Redis](https://redis.io/download) |

---

## 🔧 Installation

### Step 1: 

```bash
cd Project-BlueSky-Minds
```

### Step 2: Create Virtual Environment
cd backend
**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**You should see `(venv)` in your terminal** ✓

### Step 3: Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

Wait for installation to complete (~2-3 minutes)

### Step 4: Start Redis Server

**On Windows:**
1. Download Redis from [this link](https://github.com/microsoftarchive/redis/releases)
2. Extract and run `redis-server.exe`

**On macOS:**
```bash
brew install redis
brew services start redis
```

**On Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
```

**Verify Redis is running:**
```bash
redis-cli ping
# Should return: PONG
```

---

## ⚙️ Configuration

### Step 1: Get Your API Keys

You need two API keys (both are **FREE**):

#### 🛰️ NASA Earthdata Token
1. Go to: https://urs.earthdata.nasa.gov/users/new
2. Create a free account
3. After login, go to **Profile** → **Generate Token**
4. Copy your token (it looks like: `eyJ0eXAiOiJKV1QiLCJ...`)

#### 🌐 OpenWeather API Key
1. Go to: https://home.openweathermap.org/users/sign_up
2. Create a free account
3. Go to **API Keys** section
4. Copy your API key (32 characters)

### Step 2: Set Up Environment File

**Option A: Automated Setup (Recommended)**

```bash
# From the project root directory
chmod +x setup_env.sh
./setup_env.sh
```

This will:
- Create `.env` file automatically
- Generate secure passwords
- Prompt you to add your API keys

**Option B: Manual Setup**

```bash
# Copy the example file
cp .env.example .env

# Edit the file
nano .env
# or use your preferred text editor
```

### Step 3: Add Your API Keys

Open `.env` file and update these lines:

```bash
# Replace with your actual NASA token
NASA_EARTHDATA_TOKEN=paste_your_nasa_token_here

# Replace with your actual OpenWeather key
OPENWEATHER_API_KEY=paste_your_openweather_key_here
```

**Save the file** (Ctrl+O then Enter in nano, or Ctrl+S in most editors)

### Step 4: Secure Your .env File

**On macOS/Linux:**
```bash
chmod 600 .env
```

**On Windows:**
- Right-click `.env` → Properties → Security
- Remove all users except yourself
- Give yourself Full Control

---

## ▶️ Running the Application

### Step 1: Start the Server

```bash
# Make sure you're in the backend folder
cd backend

# Make sure virtual environment is active (you should see (venv) in terminal)

# Start the server
python -m uvicorn app.main:app --reload
```
🔧 Fix: Install Missing Dependencies
bash# Make sure you're in the backend folder with venv activated
# You should see (venv) in your prompt
```bash
pip install aiohttp
```
However, since aiohttp should be in requirements.txt, let's reinstall all dependencies properly:
bash# Reinstall all requirements to catch any missing packages
```bash
pip install -r requirements.txt --upgrade
```
🚨 If That Doesn't Work
The requirements.txt might be incomplete. Install common missing packages:
```bash
bashpip install aiohttp fastapi uvicorn redis pydantic python-dotenv requests numpy pandas
```
✅ Then Restart the Server
```bash
bashpython -m uvicorn app.main:app --reload
```
**You should see:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Application startup complete.
```

### Step 2: Verify Installation

Open a web browser and go to:
```
http://localhost:8000/docs
```

You should see the **API Documentation** page ✓

---

## 💡 Using the Application

### Method 1: Using the Web Interface

1. **Open your browser** and go to:
   ```
   http://localhost:8000
   ```

2. **Enter location details:**
   - City name or coordinates (latitude/longitude)
   - Example: New York, USA

3. **Select a date** (2-3 months in the future)
   - Example: If today is January 2025, select April 2025

4. **Click "Get Probability"**

5. **View your results:**
   - Temperature prediction with confidence level
   - Precipitation probability
   - Historical context

### Method 2: Using the API Documentation

1. **Go to the API docs:**
   ```
   http://localhost:8000/docs
   ```

2. **Find the `POST /api/weather/probability` endpoint**

3. **Click "Try it out"**

4. **Enter example data:**
   ```json
   {
     "latitude": 40.7128,
     "longitude": -74.0060,
     "target_date": "2025-04-15"
   }
   ```

5. **Click "Execute"**

6. **See the response** with weather predictions

### Method 3: Using Command Line (curl)

```bash
curl -X POST "http://localhost:8000/api/weather/probability" \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 40.7128,
    "longitude": -74.0060,
    "target_date": "2025-04-15"
  }'
```

### Method 4: Using Python Script

Create a file `test_api.py`:

```python
import requests

# API endpoint
url = "http://localhost:8000/api/weather/probability"

# Request data
payload = {
    "latitude": 40.7128,    # New York latitude
    "longitude": -74.0060,   # New York longitude
    "target_date": "2025-04-15"
}

# Make request
response = requests.post(url, json=payload)

# Print results
if response.status_code == 200:
    data = response.json()
    print(f"Temperature: {data['predictions']['temperature']['predicted']}°C")
    print(f"Confidence: {data['predictions']['temperature']['confidence']}%")
    print(f"Precipitation Probability: {data['predictions']['precipitation']['probability']*100}%")
else:
    print(f"Error: {response.status_code}")
```

Run it:
```bash
python test_api.py
```

---

## 📖 Understanding the Results

### Sample Output Explained:

```json
{
  "location": {
    "latitude": 40.7128,
    "longitude": -74.0060,
    "name": "New York"
  },
  "target_date": "2025-04-15",
  "predictions": {
    "temperature": {
      "predicted": 15.5,           ← Most likely temperature
      "confidence": 87.3,           ← How confident (87.3%)
      "range": {
        "min": 10.2,                ← Coldest likely temp
        "max": 20.8                 ← Warmest likely temp
      }
    },
    "precipitation": {
      "probability": 0.45,          ← 45% chance of rain
      "expected_mm": 12.3           ← Expected rainfall
    }
  },
  "historical_context": {
    "years_analyzed": 30            ← Based on 30 years data
  }
}
```

**What does this mean?**
- There's an **87.3% confidence** that temperature will be around **15.5°C**
- The temperature will likely be between **10.2°C and 20.8°C**
- There's a **45% chance** of precipitation (~12mm expected)
- This is based on **30 years** of historical data for this location and date

---

## 🐛 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'fastapi'"

**Solution:**
```bash
# Make sure virtual environment is activated
# You should see (venv) in your terminal

# If not activated:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Then reinstall:
cd backend
pip install -r requirements.txt
```

---

### Problem: "Redis connection refused"

**Solution:**

1. **Check if Redis is running:**
   ```bash
   redis-cli ping
   ```
   Should return `PONG`

2. **If not running, start Redis:**
   
   **Windows:** Run `redis-server.exe`
   
   **macOS:**
   ```bash
   brew services start redis
   ```
   
   **Linux:**
   ```bash
   sudo systemctl start redis-server
   ```

---

### Problem: "NASA_EARTHDATA_TOKEN is required"

**Solution:**

1. **Check if .env file exists:**
   ```bash
   ls -la .env
   ```

2. **If it doesn't exist, create it:**
   ```bash
   cp .env.example .env
   ```

3. **Add your API keys:**
   ```bash
   nano .env
   ```
   
   Update these lines:
   ```
   NASA_EARTHDATA_TOKEN=your_actual_token_here
   OPENWEATHER_API_KEY=your_actual_key_here
   ```

4. **Save and restart the server**

---

### Problem: "Port 8000 already in use"

**Solution:**

**Find what's using the port:**

**Windows:**
```bash
netstat -ano | findstr :8000
taskkill /PID <process_id> /F
```

**macOS/Linux:**
```bash
lsof -i :8000
kill -9 <process_id>
```

**Or use a different port:**
```bash
python -m uvicorn app.main:app --reload --port 8001
```

Then access at `http://localhost:8001`

---

### Problem: Server starts but API doesn't work

**Solution:**

1. **Check server logs** for errors

2. **Verify .env configuration:**
   ```bash
   cd backend
   python test_config.py
   ```

3. **Test Redis connection:**
   ```bash
   redis-cli ping
   ```

4. **Check if API keys are valid:**
   - Go to NASA Earthdata and verify token is active
   - Go to OpenWeather and check API key status

---

### Problem: Low accuracy or unexpected results

**This is normal!** Here's why:

- Project-BlueSky-Minds uses **statistical probability** based on 30-year averages
- It's designed for **long-term planning** (2-3 months ahead)
- Traditional weather apps use **real-time atmospheric data** (accurate for 1-14 days)

**Think of it this way:**
- ❌ Don't use it for: "What should I wear tomorrow?"
- ✅ Use it for: "Should I book an outdoor wedding venue in June?"

**To improve results:**
- Use for dates 2-3 months ahead (not tomorrow)
- Compare with historical patterns, not daily forecasts
- Focus on probability ranges, not exact values

---

## 🔄 Common Tasks

### Stop the Server

Press `Ctrl + C` in the terminal where the server is running

### Restart the Server

```bash
# Stop with Ctrl+C, then:
python -m uvicorn app.main:app --reload
```

### Clear Cache

```bash
redis-cli FLUSHALL
```

### Update the Application

```bash
git pull origin main
cd backend
pip install -r requirements.txt --upgrade
```

### View Logs

```bash
# Real-time logs
tail -f backend/logs/application.log

# Last 50 lines
tail -n 50 backend/logs/application.log
```

---

## 📚 Additional Resources

### API Endpoints

| Endpoint | Description | Method |
|----------|-------------|--------|
| `/api/health` | Check if server is running | GET |
| `/api/weather/probability` | Get weather probability | POST |
| `/api/location/search` | Search for locations | GET |
| `/api/weather/trends` | Get historical trends | POST |
| `/api/export` | Export data (CSV/JSON) | POST |

### Full API Documentation

Once the server is running, visit:
- **Interactive Docs:** http://localhost:8000/docs
- **Alternative Docs:** http://localhost:8000/redoc

### Example Locations to Try

| City | Latitude | Longitude |
|------|----------|-----------|
| New York, USA | 40.7128 | -74.0060 |
| London, UK | 51.5074 | -0.1278 |
| Tokyo, Japan | 35.6762 | 139.6503 |
| Sydney, Australia | -33.8688 | 151.2093 |
| Mumbai, India | 19.0760 | 72.8777 |
| Paris, France | 48.8566 | 2.3522 |

---

## 🆘 Getting Help

### Still having issues?

1. **Check the logs:**
   ```bash
   tail -f backend/logs/application.log
   ```

2. **Verify all services are running:**
   ```bash
   # Check Redis
   redis-cli ping
   
   # Check server
   curl http://localhost:8000/api/health
   ```

3. **Restart everything:**
   ```bash
   # Stop server (Ctrl+C)
   # Restart Redis
   redis-cli shutdown
   redis-server
   
   # Restart server
   cd backend
   python -m uvicorn app.main:app --reload
   ```

4. **Create an issue on GitHub:**
   - Include error messages
   - Include OS and Python version
   - Describe what you were trying to do

---

## ✅ Quick Reference Card

```bash
# START APPLICATION
cd weather-odds-pro/backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
python -m uvicorn app.main:app --reload

# STOP APPLICATION
Ctrl + C

# CHECK STATUS
curl http://localhost:8000/api/health

# VIEW DOCS
http://localhost:8000/docs

# CLEAR CACHE
redis-cli FLUSHALL

# UPDATE CODE
git pull origin main
pip install -r requirements.txt --upgrade
```

---

## 🎉 You're All Set!

Your Project-BlueSky-Minds installation is complete. Start making long-term weather predictions!

**Next Steps:**
1. Try predicting weather for your location 3 months ahead
2. Explore the API documentation
3. Compare predictions with historical data
4. Share feedback and contribute to the project

---

**Built with ❤️ for smarter
