import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    SQLALCHEMY_DATABASE_URI = 'sqlite:///timetable.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Default college timings
    DEFAULT_START_TIME = "09:00"
    DEFAULT_END_TIME = "17:00"
    DEFAULT_LUNCH_START = "13:00"
    DEFAULT_LUNCH_END = "14:00"
    
    # Lecture durations (in minutes)
    THEORY_DURATION = 60
    PRACTICAL_DURATION = 120 