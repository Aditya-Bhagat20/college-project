from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from models import db, User, Teacher, CollegeTimings, Timetable
from config import Config
import os
from datetime import datetime, timedelta
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
import json
import random

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create database tables
with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        name = request.form.get('name')
        department = request.form.get('department')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        user = User(username=username, name=name, department=department)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    teachers = Teacher.query.filter_by(hod_id=current_user.id).all()
    timings = CollegeTimings.query.filter_by(hod_id=current_user.id).first()
    return render_template('dashboard.html', teachers=teachers, timings=timings)

@app.route('/add-teacher', methods=['POST'])
@login_required
def add_teacher():
    name = request.form.get('name')
    subject = request.form.get('subject')
    lecture_type = request.form.get('lecture_type')
    
    teacher = Teacher(
        name=name,
        subject=subject,
        lecture_type=lecture_type,
        department=current_user.department,
        hod_id=current_user.id
    )
    db.session.add(teacher)
    db.session.commit()
    return redirect(url_for('dashboard'))

@app.route('/update-teacher/<int:teacher_id>', methods=['POST'])
@login_required
def update_teacher(teacher_id):
    teacher = Teacher.query.get_or_404(teacher_id)
    if teacher.hod_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    teacher.subject = request.form.get('subject')
    teacher.lecture_type = request.form.get('lecture_type')
    db.session.commit()
    return redirect(url_for('dashboard'))

@app.route('/delete-teacher/<int:teacher_id>', methods=['POST'])
@login_required
def delete_teacher(teacher_id):
    teacher = Teacher.query.get_or_404(teacher_id)
    if teacher.hod_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    db.session.delete(teacher)
    db.session.commit()
    return redirect(url_for('dashboard'))

@app.route('/update-attendance', methods=['POST'])
@login_required
def update_attendance():
    data = request.json
    teacher_id = data.get('teacher_id')
    is_present = data.get('is_present')
    
    teacher = Teacher.query.get_or_404(teacher_id)
    if teacher.hod_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    teacher.is_present = is_present
    db.session.commit()
    return jsonify({'success': True})

@app.route('/update-timings', methods=['POST'])
@login_required
def update_timings():
    timings = CollegeTimings.query.filter_by(hod_id=current_user.id).first()
    if not timings:
        timings = CollegeTimings(hod_id=current_user.id)
        db.session.add(timings)
    
    timings.start_time = request.form.get('start_time')
    timings.end_time = request.form.get('end_time')
    timings.lunch_start = request.form.get('lunch_start')
    timings.lunch_end = request.form.get('lunch_end')
    timings.short_break_start = request.form.get('short_break_start')
    timings.short_break_end = request.form.get('short_break_end')
    
    db.session.commit()
    return redirect(url_for('dashboard'))

class TimetableChromosome:
    def __init__(self, slots, teachers):
        self.slots = slots
        self.teachers = teachers
        self.schedule = {}
        self.fitness = 0
        self.initialize()

    def initialize(self):
        # Get all practical teachers
        practical_teachers = [t for t in self.teachers if t.lecture_type == 'practical']
        
        # If there are practical teachers, randomly select ONE
        if practical_teachers:
            # Shuffle practical teachers to ensure random selection
            random.shuffle(practical_teachers)
            selected_practical = practical_teachers[0]
            
            # Find all possible 2-hour slots
            possible_blocks = []
            for i in range(len(self.slots) - 1):
                slot1 = self.slots[i]
                slot2 = self.slots[i + 1]
                
                # Check if slots are consecutive (not separated by break/lunch)
                time1 = datetime.strptime(slot1, '%H:%M')
                time2 = datetime.strptime(slot2, '%H:%M')
                if (time2 - time1).total_seconds() / 3600 == 1:
                    possible_blocks.append((slot1, slot2))
            
            # If we found valid blocks, randomly select ONE
            if possible_blocks:
                # Shuffle blocks to ensure random selection
                random.shuffle(possible_blocks)
                block = possible_blocks[0]
                self.schedule[block[0]] = selected_practical
                self.schedule[block[1]] = selected_practical
        
        # Schedule theory teachers in remaining slots
        theory_teachers = [t for t in self.teachers if t.lecture_type == 'theory']
        available_teachers = theory_teachers.copy()
        random.shuffle(available_teachers)  # Shuffle for random selection
        
        for slot in self.slots:
            if slot not in self.schedule and available_teachers:
                teacher = available_teachers.pop()
                self.schedule[slot] = teacher

    def calculate_fitness(self):
        score = 0
        used_teachers = set()
        consecutive_classes = 0
        last_teacher = None
        
        # Sort slots by time
        sorted_slots = sorted(self.schedule.items(), 
                            key=lambda x: datetime.strptime(x[0], '%H:%M'))
        
        # Check practical scheduling
        practical_count = 0
        practical_blocks = {}
        for slot, teacher in sorted_slots:
            if teacher.lecture_type == 'practical':
                practical_count += 1
                if teacher not in practical_blocks:
                    practical_blocks[teacher] = [slot]
                else:
                    practical_blocks[teacher].append(slot)
        
        # Verify exactly ONE practical block
        if practical_count == 2:  # One practical in two slots
            for teacher, slots in practical_blocks.items():
                if len(slots) == 2:
                    time1 = datetime.strptime(slots[0], '%H:%M')
                    time2 = datetime.strptime(slots[1], '%H:%M')
                    if (time2 - time1).total_seconds() / 3600 == 1:
                        score += 200  # Reward proper practical block
                    else:
                        score -= 400  # Heavily penalize split practical
        else:
            score -= 400  # Penalize if not exactly one practical
        
        # Check theory scheduling
        for i, (slot, teacher) in enumerate(sorted_slots):
            if teacher.lecture_type == 'theory':
                # Check for consecutive classes
                if teacher == last_teacher:
                    consecutive_classes += 1
                    score -= 10 * consecutive_classes
                else:
                    consecutive_classes = 0
                    last_teacher = teacher
                score += 10  # Reward proper theory scheduling
            
            used_teachers.add(teacher)
        
        # Reward using all available teachers
        score += 5 * len(used_teachers)
        
        self.fitness = score
        return score

    def crossover(self, other):
        child = TimetableChromosome(self.slots, self.teachers)
        
        # Get practical blocks from both parents
        parent1_practical = None
        parent2_practical = None
        
        for slot, teacher in self.schedule.items():
            if teacher.lecture_type == 'practical':
                parent1_practical = (slot, teacher)
                break
                
        for slot, teacher in other.schedule.items():
            if teacher.lecture_type == 'practical':
                parent2_practical = (slot, teacher)
                break
        
        # Randomly choose which parent's practical to use
        if parent1_practical and parent2_practical:
            if random.random() < 0.5:
                practical = parent1_practical
            else:
                practical = parent2_practical
        elif parent1_practical:
            practical = parent1_practical
        elif parent2_practical:
            practical = parent2_practical
        
        # If we have a practical, schedule it
        if practical:
            slot, teacher = practical
            # Find the next slot for this practical
            slot_index = self.slots.index(slot)
            if slot_index + 1 < len(self.slots):
                next_slot = self.slots[slot_index + 1]
                child.schedule[slot] = teacher
                child.schedule[next_slot] = teacher
        
        # Fill remaining slots with theory teachers
        theory_teachers = [t for t in self.teachers if t.lecture_type == 'theory']
        available_teachers = theory_teachers.copy()
        random.shuffle(available_teachers)  # Shuffle for random selection
        
        for slot in self.slots:
            if slot not in child.schedule and available_teachers:
                teacher = available_teachers.pop()
                child.schedule[slot] = teacher
        
        return child

    def mutate(self, mutation_rate=0.1):
        if random.random() < mutation_rate:
            # Only mutate theory teachers, keep practical block intact
            theory_slots = [slot for slot, teacher in self.schedule.items() 
                          if teacher.lecture_type == 'theory']
            if len(theory_slots) >= 2:
                slot1, slot2 = random.sample(theory_slots, 2)
                self.schedule[slot1], self.schedule[slot2] = self.schedule[slot2], self.schedule[slot1]

def generate_timetable_slots(timings):
    slots = []
    current_time = datetime.strptime(timings.start_time, '%H:%M')
    end_time = datetime.strptime(timings.end_time, '%H:%M')
    lunch_start = datetime.strptime(timings.lunch_start, '%H:%M')
    lunch_end = datetime.strptime(timings.lunch_end, '%H:%M')
    
    while current_time < end_time:
        if current_time >= lunch_start and current_time < lunch_end:
            current_time = lunch_end
            continue
            
        if timings.short_break_start and timings.short_break_end:
            break_start = datetime.strptime(timings.short_break_start, '%H:%M')
            break_end = datetime.strptime(timings.short_break_end, '%H:%M')
            if current_time >= break_start and current_time < break_end:
                current_time = break_end
                continue
        
        slots.append(current_time.strftime('%H:%M'))
        current_time += timedelta(minutes=60)
    
    return slots

def genetic_algorithm(slots, teachers, population_size=50, generations=100):
    # Initialize population
    population = [TimetableChromosome(slots, teachers) for _ in range(population_size)]
    
    for generation in range(generations):
        # Calculate fitness for all chromosomes
        for chromosome in population:
            chromosome.calculate_fitness()
        
        # Sort population by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Select top 50% for next generation
        next_generation = population[:population_size//2]
        
        # Create offspring through crossover
        while len(next_generation) < population_size:
            parent1 = random.choice(population[:population_size//2])
            parent2 = random.choice(population[:population_size//2])
            child = parent1.crossover(parent2)
            child.mutate()
            next_generation.append(child)
        
        population = next_generation
    
    # Return best solution
    return max(population, key=lambda x: x.fitness)

@app.route('/generate-timetable', methods=['POST'])
@login_required
def generate_timetable():
    # Clear existing timetable for the day
    Timetable.query.filter_by(
        hod_id=current_user.id,
        date=datetime.now().date()
    ).delete()
    
    timings = CollegeTimings.query.filter_by(hod_id=current_user.id).first()
    if not timings:
        return jsonify({'error': 'Please set college timings first'}), 400
    
    teachers = Teacher.query.filter_by(
        hod_id=current_user.id,
        is_present=True
    ).all()
    
    if not teachers:
        return jsonify({'error': 'No teachers marked present today'}), 400
    
    # Generate time slots
    slots = generate_timetable_slots(timings)
    if not slots:
        return jsonify({'error': 'No valid time slots available'}), 400
    
    # Run genetic algorithm
    best_solution = genetic_algorithm(slots, teachers)
    
    # Convert solution to timetable format
    timetable = []
    for slot, teacher in best_solution.schedule.items():
        end_time = (datetime.strptime(slot, '%H:%M') + timedelta(minutes=60)).strftime('%H:%M')
        timetable.append({
            'teacher_id': teacher.id,
            'teacher_name': teacher.name,
            'start_time': slot,
            'end_time': end_time,
            'subject': teacher.subject,
            'lecture_type': teacher.lecture_type
        })
    
    # Sort timetable by start time
    timetable.sort(key=lambda x: datetime.strptime(x['start_time'], '%H:%M'))
    
    # Save generated timetable
    for entry in timetable:
        db.session.add(Timetable(
            hod_id=current_user.id,
            date=datetime.now().date(),
            teacher_id=entry['teacher_id'],
            start_time=entry['start_time'],
            end_time=entry['end_time'],
            subject=entry['subject'],
            lecture_type=entry['lecture_type']
        ))
    
    db.session.commit()
    return jsonify({'success': True, 'timetable': timetable})

@app.route('/download-timetable')
@login_required
def download_timetable():
    # Get today's timetable
    timetable = Timetable.query.filter_by(
        hod_id=current_user.id,
        date=datetime.now().date()
    ).all()
    
    if not timetable:
        flash('No timetable generated for today')
        return redirect(url_for('dashboard'))
    
    # Create PDF
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    
    # Add header
    p.setFont("Helvetica-Bold", 16)
    p.drawString(100, 750, f"Timetable for {current_user.department} Department")
    p.drawString(100, 730, f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    
    # Add table header
    p.setFont("Helvetica-Bold", 12)
    p.drawString(100, 700, "Time")
    p.drawString(200, 700, "Subject")
    p.drawString(300, 700, "Teacher")
    p.drawString(400, 700, "Type")
    
    # Add table content
    p.setFont("Helvetica", 12)
    y = 680
    for entry in timetable:
        teacher = Teacher.query.get(entry.teacher_id)
        p.drawString(100, y, f"{entry.start_time} - {entry.end_time}")
        p.drawString(200, y, entry.subject)
        p.drawString(300, y, teacher.name)
        p.drawString(400, y, entry.lecture_type)
        y -= 20
    
    p.save()
    buffer.seek(0)
    
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"timetable_{datetime.now().strftime('%Y-%m-%d')}.pdf",
        mimetype='application/pdf'
    )

if __name__ == '__main__':
    app.run(debug=True) 