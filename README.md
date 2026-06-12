# FastAPI Regression Docker App

A Dockerized machine learning application consisting of a FastAPI backend and a Streamlit frontend. The application allows users to predict values using a linear regression model, retrain the model with new data points, and visualize training data together with the regression line.

## Run locally
### Backend
```bash
uvicorn app:app --host 127.0.0.1 --port 8008 --reload
```

### Frontend
```bash
streamlit run app/streamlit_app.py
```

## Run with Docker

```bash
docker build -t app-fastapi-ml .
docker run --name fastapi-ml -e PORT=8008 -p 8008:8008 -d app-fastapi-ml
```

Backend:
http://localhost:8008/docs

### Run frontend
```bash
streamlit run app/streamlit_app.py
```