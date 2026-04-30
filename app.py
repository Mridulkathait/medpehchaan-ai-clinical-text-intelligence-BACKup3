import os
from ui import run_app

if __name__ == "__main__":
    # Set port for Render deployment
    port = int(os.environ.get("PORT", 8501))
    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"
    run_app()
