# nlp-factcheck-enhanced
# NLP Phase-wise Analysis with Google Fact Check

Enhanced version of the NLP model comparison app with Google Fact Check API integration.

## What's New?

This enhanced version adds:
- **Google Fact Check API Integration**: Automatically verify statements using Google's Fact Check Tools
- **Dual Comparison**: Compare model performance against both user-provided labels AND Google Fact Check results

## Prerequisites

- Python 3.8 or higher
- A Google Cloud account (for Fact Check API)
- Git installed on your computer

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
4. Enter your Google API key in the sidebar (Use this key: AIzaSyAu-jqo6Gz1Ubea3bj6NJShmKnCHoL6ZhI )
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

## Understanding the NLP Phases

- **Lexical & Morphological**: Word-level analysis (tokenization, lemmatization)
- **Syntactic**: Grammatical structure (parts of speech)
- **Semantic**: Meaning and sentiment
- **Discourse**: Sentence structure and flow
- **Pragmatic**: Context and intent (modals, questions)

## Need Help?

Common questions:
1. **Is the API free?** Google provides a free tier with generous limits
2. **How many requests can I make?** Check Google Cloud quotas, typically 10,000/day free



