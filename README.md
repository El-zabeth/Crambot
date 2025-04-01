# Crambot

## Install

```bash
python -m venv venv
./venv/Scripts/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Create a .env file based on the .env.example and populate with you own API Key

## Run

```bash
streamlit run app.py
```