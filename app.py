# Top-level entrypoint for Streamlit
import os
import sys

app_path = os.path.join(os.path.dirname(__file__), 'app', 'app.py')
with open(app_path, 'r', encoding='utf-8') as f:
    code = f.read()
exec(code)
