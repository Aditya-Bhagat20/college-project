from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    name = db.Column(db.String(100), nullable=False)
    department = db.Column(db.String(100), nullable=False)
    is_hod = db.Column(db.Boolean, default=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Teacher(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    subject = db.Column(db.String(100), nullable=False)
    lecture_type = db.Column(db.String(20), nullable=False)  # 'theory' or 'practical'
    department = db.Column(db.String(100), nullable=False)
    hod_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    is_present = db.Column(db.Boolean, default=False)

class CollegeTimings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    hod_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    start_time = db.Column(db.String(5), nullable=False)  # Format: "HH:MM"
    end_time = db.Column(db.String(5), nullable=False)
    lunch_start = db.Column(db.String(5), nullable=False)
    lunch_end = db.Column(db.String(5), nullable=False)
    short_break_start = db.Column(db.String(5), nullable=True)
    short_break_end = db.Column(db.String(5), nullable=True)

class Timetable(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    hod_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    teacher_id = db.Column(db.Integer, db.ForeignKey('teacher.id'), nullable=False)
    start_time = db.Column(db.String(5), nullable=False)
    end_time = db.Column(db.String(5), nullable=False)
    subject = db.Column(db.String(100), nullable=False)
    lecture_type = db.Column(db.String(20), nullable=False) 