python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
get .env file from the CEO
uvicorn main:app --port 2020

Don't forget to have a redis server running on port 6379 and a postgres server running on port 5432 through docker-compose up -d
