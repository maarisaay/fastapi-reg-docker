http://127.0.0.1:8008/docs#/

Docker
docker build -t app-fastapi-ml .
docker run --name fastapi-ml -e PORT=8008 -p 8008:8008 -d app-fastapi-ml