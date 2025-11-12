# nlp-factcheck-enhanced
# NLP Phase-wise Analysis with Google Fact Check

Enhanced version of the NLP model comparison app with Google Fact Check API integration.

## What's New?

This enhanced version adds:
- **Google Fact Check API Integration**: Automatically verify statements using Google's Fact Check Tools
- **Dual Comparison**: Compare model performance against both user-provided labels AND Google Fact Check results
- **Enhanced Metrics**: Added Precision, Recall, and F1-Score alongside Accuracy
- **Better Visualizations**: Side-by-side comparison charts

## Prerequisites

- Python 3.8 or higher
- A Google Cloud account (for Fact Check API)
- Git installed on your computer

## Getting Your Google Fact Check API Key

### Step 1: Create a Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click on the project dropdown at the top
3. Click "New Project"
4. Give it a name (e.g., "nlp-factcheck-project")
5. Click "Create"

### Step 2: Enable Fact Check Tools API
1. In the search bar, type "Fact Check Tools API"
2. Click on it and press "Enable"
3. Wait for it to be enabled

### Step 3: Create API Key
1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "API Key"
3. Copy the API key that appears
4. **Important**: Restrict your API key:
   - Click on the key you just created
   - Under "API restrictions", select "Restrict key"
   - Choose "Fact Check Tools API"
   - Save

## Local Setup and Testing

### Step 1: Download the Files
1. Download all these files to a folder on your computer:
   - `main_enhanced.py`
   - `requirements.txt`
   - `README.md` (this file)

### Step 2: Install Python Packages
Open your terminal/command prompt and navigate to the folder:

```bash
cd path/to/your/folder
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### Step 3: Run Locally
Run the Streamlit app:

```bash
streamlit run main_enhanced.py
```

The app will open in your browser at `http://localhost:8501`

## Deploying to Streamlit Cloud

### Step 1: Create a GitHub Account
1. Go to [github.com](https://github.com)
2. Sign up for a free account if you don't have one

### Step 2: Create a New Repository
1. Click the "+" icon in the top right
2. Select "New repository"
3. Name it (e.g., "nlp-factcheck-enhanced")
4. Make it **Public**
5. Check "Add a README file"
6. Click "Create repository"

### Step 3: Upload Your Files
1. In your new repository, click "Add file" > "Upload files"
2. Drag and drop these files:
   - `main_enhanced.py`
   - `requirements.txt`
   - `README.md`
3. Click "Commit changes"

### Step 4: Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository (nlp-factcheck-enhanced)
5. Main file path: `main_enhanced.py`
6. Click "Deploy!"

Your app will be live in a few minutes! 🎉

## How to Use the App

### 1. Prepare Your Data
Your CSV file should have:
- A column with text statements (e.g., "The Earth is flat")
- A column with binary labels: 
  - 1/0 or True/False or TRUE/FALSE

Example CSV:
```csv
statement,label
The Earth revolves around the Sun,1
Water boils at 50 degrees Celsius,0
Humans need oxygen to breathe,1
```

### 2. Run the Analysis
1. Upload your CSV file
2. Select the text column (statements)
3. Select the target column (labels)
4. Enter your Google API key in the sidebar
5. Choose an NLP phase
6. Click "Run Model Comparison"

### 3. Interpret Results
You'll see two sections:
- **User-Provided Labels**: How models perform against your labels
- **Google Fact Check Labels**: How models perform against Google's fact-checking

Compare both to:
- Validate your labels
- Understand model reliability
- Identify potential labeling errors

## Troubleshooting

### Issue: "API Key is invalid"
- Check that you copied the entire key
- Ensure the Fact Check Tools API is enabled
- Verify the key restrictions allow Fact Check Tools API

### Issue: "No data returned from Google Fact Check"
- The statements might be too specific or uncommon
- Google Fact Check primarily covers well-known claims
- Try with more mainstream factual statements

### Issue: "Module not found error"
- Run: `pip install -r requirements.txt` again
- Make sure you're in the correct folder

### Issue: Streamlit app won't start
- Check Python version: `python --version` (should be 3.8+)
- Try: `python -m streamlit run main_enhanced.py`

## Understanding the NLP Phases

- **Lexical & Morphological**: Word-level analysis (tokenization, lemmatization)
- **Syntactic**: Grammatical structure (parts of speech)
- **Semantic**: Meaning and sentiment
- **Discourse**: Sentence structure and flow
- **Pragmatic**: Context and intent (modals, questions)

## Metrics Explained

- **Accuracy**: Overall correct predictions
- **Precision**: How many predicted positives are actually positive
- **Recall**: How many actual positives were found
- **F1-Score**: Balance between precision and recall

## Need Help?

Common questions:
1. **Can I use this without an API key?** Yes! The app will work with user-provided labels only
2. **Is the API free?** Google provides a free tier with generous limits
3. **How many requests can I make?** Check Google Cloud quotas, typically 10,000/day free
4. **Can I edit the code?** Absolutely! The code is open and customizable

## License

This project is for educational purposes. Feel free to modify and use for your assignments!

## Assignment Tips

For your master's course:
1. **Compare results**: Discuss differences between user labels and Google Fact Check
2. **Analyze patterns**: Which models perform better on which types of statements?
3. **Document limitations**: What types of claims does Google Fact Check cover?
4. **Suggest improvements**: How could the comparison be enhanced?

Good luck with your assignment! 🎓
