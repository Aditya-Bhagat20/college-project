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
    def __init__(self, slots, teachers, division_id=None, existing_schedules=None):
        self.slots = slots
        self.teachers = teachers
        self.division_id = division_id
        self.existing_schedules = existing_schedules or {}  # Track other divisions' schedules
        self.schedule = {}
        self.fitness = 0
        self.initialize()

    def initialize(self):
        # Get all practical teachers
        practical_teachers = [t for t in self.teachers if t.lecture_type == 'practical']
        theory_teachers = [t for t in self.teachers if t.lecture_type == 'theory']
        
        # Initialize teacher availability
        teacher_availability = {teacher.id: True for teacher in self.teachers}
        
        # If there are practical teachers, schedule them first
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
                    # Check if teacher is available in this block across all divisions
                    if not self.is_teacher_busy(selected_practical.id, slot1) and \
                       not self.is_teacher_busy(selected_practical.id, slot2):
                        possible_blocks.append((slot1, slot2))
            
            # If we found valid blocks, randomly select ONE
            if possible_blocks:
                random.shuffle(possible_blocks)
                block = possible_blocks[0]
                self.schedule[block[0]] = selected_practical
                self.schedule[block[1]] = selected_practical
                teacher_availability[selected_practical.id] = False
        
        # Schedule theory teachers in remaining slots
        available_teachers = theory_teachers.copy()
        random.shuffle(available_teachers)
        
        # Track teacher load to ensure fair distribution
        teacher_load = {teacher.id: 0 for teacher in theory_teachers}
        max_load = len(self.slots) // len(theory_teachers) + 1
        
        # Sort slots to ensure better distribution
        sorted_slots = sorted(self.slots, key=lambda x: datetime.strptime(x, '%H:%M'))
        
        for slot in sorted_slots:
            if slot not in self.schedule:
                # Find an available teacher with the least load
                available_teachers.sort(key=lambda t: (teacher_load[t.id], -len(self.get_available_slots(t.id))))
                for teacher in available_teachers:
                    if (teacher_availability[teacher.id] and 
                        teacher_load[teacher.id] < max_load and
                        not self.is_teacher_busy(teacher.id, slot)):
                        self.schedule[slot] = teacher
                        teacher_availability[teacher.id] = False
                        teacher_load[teacher.id] += 1
                        break

    def is_teacher_busy(self, teacher_id, slot):
        """Check if teacher is already scheduled in this slot in any division"""
        for div_schedule in self.existing_schedules.values():
            if slot in div_schedule and div_schedule[slot].id == teacher_id:
                return True
        return False

    def get_available_slots(self, teacher_id):
        """Get list of slots where teacher is available"""
        return [slot for slot in self.slots 
                if slot not in self.schedule and not self.is_teacher_busy(teacher_id, slot)]

    def calculate_fitness(self):
        score = 0
        teacher_slots = {}  # Track slots for each teacher
        consecutive_classes = {}  # Track consecutive classes for each teacher
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
        
        # Check theory scheduling and teacher collisions
        for i, (slot, teacher) in enumerate(sorted_slots):
            if teacher.lecture_type == 'theory':
                # Track teacher slots
                if teacher.id not in teacher_slots:
                    teacher_slots[teacher.id] = []
                teacher_slots[teacher.id].append(slot)
                
                # Check for consecutive classes
                if teacher == last_teacher:
                    if teacher.id not in consecutive_classes:
                        consecutive_classes[teacher.id] = 1
                    consecutive_classes[teacher.id] += 1
                    score -= 30 * consecutive_classes[teacher.id]  # Increased penalty for consecutive classes
                else:
                    consecutive_classes[teacher.id] = 0
                    last_teacher = teacher
                
                score += 10  # Reward proper theory scheduling
            
            # Check for teacher collisions
            for other_teacher_id, other_slots in teacher_slots.items():
                if other_teacher_id != teacher.id:
                    if slot in other_slots:
                        score -= 2000  # Heavily penalize teacher collisions
        
        # Check for collisions with other divisions
        for slot, teacher in self.schedule.items():
            if self.is_teacher_busy(teacher.id, slot):
                score -= 3000  # Even heavier penalty for cross-division collisions
        
        # Reward using all available teachers
        score += 5 * len(teacher_slots)
        
        # Penalize unused teachers
        unused_teachers = len(self.teachers) - len(teacher_slots)
        score -= 200 * unused_teachers  # Increased penalty for unused teachers
        
        # Check teacher load distribution
        teacher_loads = [len(slots) for slots in teacher_slots.values()]
        if teacher_loads:
            avg_load = sum(teacher_loads) / len(teacher_loads)
            for load in teacher_loads:
                if abs(load - avg_load) > 1:  # If load is significantly different from average
                    score -= 100 * abs(load - avg_load)  # Increased penalty for uneven distribution
        
        self.fitness = score
        return score

    def crossover(self, other):
        child = TimetableChromosome(self.slots, self.teachers, self.division_id, self.existing_schedules)
        
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
                if not self.is_teacher_busy(teacher.id, slot) and not self.is_teacher_busy(teacher.id, next_slot):
                    child.schedule[slot] = teacher
                    child.schedule[next_slot] = teacher
        
        # Fill remaining slots with theory teachers using both parents
        theory_teachers = [t for t in self.teachers if t.lecture_type == 'theory']
        available_teachers = theory_teachers.copy()
        random.shuffle(available_teachers)
        
        # Track used teachers and their loads
        used_teachers = set()
        teacher_load = {teacher.id: 0 for teacher in theory_teachers}
        max_load = len(self.slots) // len(theory_teachers) + 1
        
        for slot in self.slots:
            if slot not in child.schedule:
                # Try to get a teacher from either parent
                parent1_teacher = self.schedule.get(slot)
                parent2_teacher = other.schedule.get(slot)
                
                # Choose a teacher that hasn't been used too much and isn't busy
                if (parent1_teacher and parent1_teacher.id not in used_teachers and 
                    teacher_load[parent1_teacher.id] < max_load and
                    not self.is_teacher_busy(parent1_teacher.id, slot)):
                    child.schedule[slot] = parent1_teacher
                    used_teachers.add(parent1_teacher.id)
                    teacher_load[parent1_teacher.id] += 1
                elif (parent2_teacher and parent2_teacher.id not in used_teachers and 
                      teacher_load[parent2_teacher.id] < max_load and
                      not self.is_teacher_busy(parent2_teacher.id, slot)):
                    child.schedule[slot] = parent2_teacher
                    used_teachers.add(parent2_teacher.id)
                    teacher_load[parent2_teacher.id] += 1
                else:
                    # If no available teacher from parents, use a random available teacher
                    available_teachers.sort(key=lambda t: (teacher_load[t.id], -len(self.get_available_slots(t.id))))
                    for teacher in available_teachers:
                        if (teacher.id not in used_teachers and 
                            teacher_load[teacher.id] < max_load and
                            not self.is_teacher_busy(teacher.id, slot)):
                            child.schedule[slot] = teacher
                            used_teachers.add(teacher.id)
                            teacher_load[teacher.id] += 1
                            break
        
        return child

    def mutate(self, mutation_rate=0.3):
        if random.random() < mutation_rate:
            # Only mutate theory teachers, keep practical block intact
            theory_slots = [slot for slot, teacher in self.schedule.items() 
                          if teacher.lecture_type == 'theory']
            
            if len(theory_slots) >= 2:
                # Get two random slots
                slot1, slot2 = random.sample(theory_slots, 2)
                
                # Get the teachers
                teacher1 = self.schedule[slot1]
                teacher2 = self.schedule[slot2]
                
                # Swap teachers only if they're different and it won't cause collisions
                if teacher1 != teacher2:
                    # Check if the swap would cause consecutive classes or collisions
                    time1 = datetime.strptime(slot1, '%H:%M')
                    time2 = datetime.strptime(slot2, '%H:%M')
                    if (abs((time2 - time1).total_seconds() / 3600) > 1 and  # Not consecutive
                        not self.is_teacher_busy(teacher1.id, slot2) and     # No collision for teacher1
                        not self.is_teacher_busy(teacher2.id, slot1)):       # No collision for teacher2
                        self.schedule[slot1] = teacher2
                        self.schedule[slot2] = teacher1

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

def genetic_algorithm(slots, teachers, division_id=None, existing_schedules=None, population_size=300, generations=500):
    try:
        app.logger.info(f"Starting genetic algorithm for division {division_id}")
        app.logger.info(f"Population size: {population_size}, Generations: {generations}")
        app.logger.info(f"Number of slots: {len(slots)}, Number of teachers: {len(teachers)}")
        
        if not slots or not teachers:
            app.logger.error("Invalid input: slots or teachers list is empty")
            return None
            
        # Initialize population
        try:
            population = []
            for i in range(population_size):
                try:
                    chromosome = TimetableChromosome(slots, teachers, division_id, existing_schedules)
                    chromosome.calculate_fitness()
                    population.append(chromosome)
                except Exception as e:
                    app.logger.error(f"Error creating chromosome {i}: {str(e)}")
                    continue
            
            if not population:
                app.logger.error("Failed to create any valid chromosomes")
                return None
                
        except Exception as e:
            app.logger.error(f"Error initializing population: {str(e)}")
            return None
        
        best_fitness = float('-inf')
        no_improvement_count = 0
        best_solution = None
        
        for generation in range(generations):
            try:
                # Calculate fitness for all chromosomes
                for chromosome in population:
                    try:
                        chromosome.calculate_fitness()
                    except Exception as e:
                        app.logger.error(f"Error calculating fitness: {str(e)}")
                        continue
                
                # Sort population by fitness
                population.sort(key=lambda x: x.fitness, reverse=True)
                
                # Keep track of best solution
                if population[0].fitness > best_fitness:
                    best_fitness = population[0].fitness
                    best_solution = population[0]
                    no_improvement_count = 0
                    app.logger.info(f"New best fitness found: {best_fitness} in generation {generation}")
                else:
                    no_improvement_count += 1
                
                # Early stopping if no improvement for too long
                if no_improvement_count > 100:
                    app.logger.info(f"Early stopping at generation {generation} due to no improvement")
                    break
                
                # Select top 50% for next generation
                next_generation = population[:population_size//2]
                
                # Create offspring through crossover
                while len(next_generation) < population_size:
                    try:
                        parent1 = random.choice(population[:population_size//2])
                        parent2 = random.choice(population[:population_size//2])
                        child = parent1.crossover(parent2)
                        child.mutate(mutation_rate=0.3)
                        child.calculate_fitness()
                        next_generation.append(child)
                    except Exception as e:
                        app.logger.error(f"Error creating offspring: {str(e)}")
                        continue
                
                population = next_generation
                
                # Early stopping if we have a good solution
                if population[0].fitness > 2000:
                    app.logger.info(f"Early stopping at generation {generation} due to good solution")
                    break
                    
            except Exception as e:
                app.logger.error(f"Error in generation {generation}: {str(e)}")
                continue
        
        if not best_solution:
            app.logger.error("No valid solution found after all generations")
            return None
            
        app.logger.info(f"Genetic algorithm completed for division {division_id}")
        app.logger.info(f"Best fitness achieved: {best_fitness}")
        return best_solution
        
    except Exception as e:
        app.logger.error(f"Critical error in genetic algorithm: {str(e)}")
        app.logger.error(f"Error type: {type(e).__name__}")
        return None

@app.route('/generate-timetable', methods=['POST'])
@login_required
def generate_timetable():
    try:
        # Get number of divisions from request
        data = request.get_json()
        if not data:
            app.logger.error("Invalid request data: No JSON data received")
            return jsonify({'error': 'Invalid request data'}), 400
            
        num_divisions = data.get('num_divisions', 1)
        if not isinstance(num_divisions, int) or num_divisions < 1:
            app.logger.error(f"Invalid number of divisions: {num_divisions}")
            return jsonify({'error': 'Invalid number of divisions'}), 400
        
        # Clear existing timetable for the day
        try:
            Timetable.query.filter_by(
                hod_id=current_user.id,
                date=datetime.now().date()
            ).delete()
            db.session.commit()
        except Exception as e:
            app.logger.error(f"Error clearing existing timetable: {str(e)}")
            return jsonify({'error': 'Error clearing existing timetable'}), 500
        
        timings = CollegeTimings.query.filter_by(hod_id=current_user.id).first()
        if not timings:
            app.logger.error("College timings not set")
            return jsonify({'error': 'Please set college timings first'}), 400
        
        teachers = Teacher.query.filter_by(
            hod_id=current_user.id,
            is_present=True
        ).all()
        
        if not teachers:
            app.logger.error("No teachers marked present today")
            return jsonify({'error': 'No teachers marked present today'}), 400
        
        app.logger.info(f"Found {len(teachers)} teachers present for {num_divisions} divisions")
        
        # Generate time slots
        slots = generate_timetable_slots(timings)
        if not slots:
            app.logger.error("No valid time slots available")
            return jsonify({'error': 'No valid time slots available'}), 400
        
        app.logger.info(f"Generated {len(slots)} time slots")
        
        # Generate timetables for each division
        all_timetables = []
        existing_schedules = {}  # Track schedules of other divisions
        
        for division in range(num_divisions):
            try:
                app.logger.info(f"Generating timetable for division {division + 1}")
                
                # Run genetic algorithm for this division
                best_solution = genetic_algorithm(slots, teachers, division + 1, existing_schedules)
                
                if not best_solution:
                    app.logger.error(f"No valid solution found for division {division + 1}")
                    return jsonify({'error': f'No valid solution found for division {division + 1}'}), 500
                
                if not best_solution.schedule:
                    app.logger.error(f"Empty schedule generated for division {division + 1}")
                    return jsonify({'error': f'Empty schedule generated for division {division + 1}'}), 500
                
                # Store this division's schedule
                existing_schedules[division + 1] = best_solution.schedule
                
                # Convert solution to timetable format
                timetable = []
                for slot, teacher in best_solution.schedule.items():
                    try:
                        end_time = (datetime.strptime(slot, '%H:%M') + timedelta(minutes=60)).strftime('%H:%M')
                        timetable.append({
                            'teacher_id': teacher.id,
                            'teacher_name': teacher.name,
                            'start_time': slot,
                            'end_time': end_time,
                            'subject': teacher.subject,
                            'lecture_type': teacher.lecture_type
                        })
                    except Exception as e:
                        app.logger.error(f"Error processing slot {slot} for teacher {teacher.name}: {str(e)}")
                        continue
                
                if not timetable:
                    app.logger.error(f"Empty timetable generated for division {division + 1}")
                    return jsonify({'error': f'Empty timetable generated for division {division + 1}'}), 500
                
                # Sort timetable by start time
                timetable.sort(key=lambda x: datetime.strptime(x['start_time'], '%H:%M'))
                all_timetables.append(timetable)
                
                # Save generated timetable
                for entry in timetable:
                    try:
                        db.session.add(Timetable(
                            hod_id=current_user.id,
                            date=datetime.now().date(),
                            teacher_id=entry['teacher_id'],
                            start_time=entry['start_time'],
                            end_time=entry['end_time'],
                            subject=entry['subject'],
                            lecture_type=entry['lecture_type'],
                            division=division + 1
                        ))
                    except Exception as e:
                        app.logger.error(f"Error saving timetable entry: {str(e)}")
                        db.session.rollback()
                        return jsonify({'error': f'Error saving timetable entry: {str(e)}'}), 500
                
                try:
                    db.session.commit()
                except Exception as e:
                    app.logger.error(f"Error committing timetable for division {division + 1}: {str(e)}")
                    db.session.rollback()
                    return jsonify({'error': f'Error saving timetable for division {division + 1}'}), 500
                
                app.logger.info(f"Successfully generated timetable for division {division + 1}")
                
            except Exception as e:
                app.logger.error(f"Error generating timetable for division {division + 1}: {str(e)}")
                app.logger.error(f"Error details: {type(e).__name__}: {str(e)}")
                db.session.rollback()
                return jsonify({'error': f'Error generating timetable for division {division + 1}: {str(e)}'}), 500
        
        return jsonify({'success': True, 'timetables': all_timetables})
        
    except Exception as e:
        app.logger.error(f"Error in generate_timetable: {str(e)}")
        app.logger.error(f"Error details: {type(e).__name__}: {str(e)}")
        db.session.rollback()
        return jsonify({'error': f'An error occurred while generating timetable: {str(e)}'}), 500

@app.route('/download-timetable')
@login_required
def download_timetable():
    # Get today's timetable
    timetable = Timetable.query.filter_by(
        hod_id=current_user.id,
        date=datetime.now().date()
    ).order_by(Timetable.division, Timetable.start_time).all()
    
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
    
    y = 700
    current_division = None
    
    for entry in timetable:
        if current_division != entry.division:
            current_division = entry.division
            y -= 30
            p.setFont("Helvetica-Bold", 14)
            p.drawString(100, y, f"Division {current_division}")
            y -= 20
            
            # Add table header
            p.setFont("Helvetica-Bold", 12)
            p.drawString(100, y, "Time")
            p.drawString(200, y, "Subject")
            p.drawString(300, y, "Teacher")
            p.drawString(400, y, "Type")
            y -= 20
        
        # Add table content
        p.setFont("Helvetica", 12)
        teacher = Teacher.query.get(entry.teacher_id)
        p.drawString(100, y, f"{entry.start_time} - {entry.end_time}")
        p.drawString(200, y, entry.subject)
        p.drawString(300, y, teacher.name)
        p.drawString(400, y, entry.lecture_type)
        y -= 20
        
        # Add page break if needed
        if y < 50:
            p.showPage()
            y = 750
            p.setFont("Helvetica-Bold", 16)
            p.drawString(100, y, f"Timetable for {current_user.department} Department (Continued)")
            y -= 30
    
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