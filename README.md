### Network Security Project for Phishing Data

This project now includes a Streamlit interface for a stronger demo-ready UI.

#### Run the Streamlit app

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

#### Run the FastAPI backend

```bash
uvicorn app:app --reload
```

#### What the Streamlit UI includes

- Dashboard with dataset summary
- Single phishing prediction form
- Batch CSV upload and downloadable results
- Feature guide for project presentation
