import os
import requests
import pandas as pd
import streamlit as st
import tempfile


def download_file_from_google_drive(file_id):
    """Download large Google Drive file (>100MB) handling confirmation token."""
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)

    # Check for confirmation token for large files
    def get_confirm_token(resp):
        for key, value in resp.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    token = get_confirm_token(response)
    if token:
        response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)

    # Write to a temporary file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")

    CHUNK_SIZE = 32768
    for chunk in response.iter_content(CHUNK_SIZE):
        if chunk:
            tmp.write(chunk)

    tmp.flush()
    return tmp.name


@st.cache_data
def load_crime_data():
    """Load large dataset from Google Drive or fallback local."""
    try:
        # file IDs from environment variables
        file_id = os.getenv("GD_CLEAN_FILE_ID", "").strip()

        if not file_id:
            st.error("❌ No Google Drive file ID provided. Set GD_CLEAN_FILE_ID in Streamlit Secrets.")
            return None, None, None

        st.info("📥 Downloading dataset from Google Drive... (only first time)")

        # Download CSV
        downloaded_path = download_file_from_google_drive(file_id)

        df = pd.read_csv(downloaded_path, parse_dates=['datetime', 'date'], low_memory=False)

        # Clean & format dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df.columns = [c.strip() for c in df.columns]

        if 'day_name' not in df.columns:
            df['day_name'] = df['date'].dt.day_name()
        if 'is_weekend' not in df.columns:
            df['is_weekend'] = df['date'].dt.weekday >= 5

        return df, df['date'].min(), df['date'].max()

    except Exception as e:
        st.error(f"❌ Error loading Google Drive file: {str(e)}")
        return None, None, None
