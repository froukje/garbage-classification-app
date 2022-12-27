FROM python:3.9
WORKDIR app/
COPY ["streamlit_app.py", "garbage.jpg", "./"]
COPY requirements.txt .
RUN pip install -r requirements.txt
#ENTRYPOINT ["streamlit", "run", "streamlit_app.py"]
