import pandas as pd
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv
load_dotenv()

class SchedulingTools:
    """
    Enhanced Tools and utilities for the AI Scheduling Agent with improved email functionality.
    """
    
    def __init__(self):
        """Initialize paths to data files."""
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        self.patients_file = os.path.join(data_dir, 'patients.csv')
        self.schedule_file = os.path.join(data_dir, 'doctor_schedules.csv')
        self.appointments_file = os.path.join(data_dir, 'appointments_report.csv')
        self._ensure_data_files_exist()

    def _ensure_data_files_exist(self):
        """Create data files if they don't exist."""
        os.makedirs(os.path.dirname(self.patients_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.schedule_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.appointments_file), exist_ok=True)
        
        if not os.path.exists(self.patients_file):
            self._create_sample_patients()
        if not os.path.exists(self.schedule_file):
            self._create_doctor_schedule()
        if not os.path.exists(self.appointments_file):
            self._create_appointments_file()

    def _create_sample_patients(self):
        """Create sample patient data."""
        try:
            from faker import Faker
            fake = Faker()
            # fake.seed(42)
            
            data = []
            for i in range(50):
                data.append({
                    'patient_id': i + 1,
                    'first_name': fake.first_name(),
                    'last_name': fake.last_name(),
                    'dob': fake.date_of_birth(minimum_age=18, maximum_age=90).strftime('%Y-%m-%d'),
                    'is_returning': bool(i % 2),
                    'email': fake.email(),
                    'location': fake.address().replace('\n', ', '),
                    'phone_number': fake.phone_number(),
                    'insurance_carrier': fake.random_element(elements=(
                        'Blue Cross Blue Shield', 'Aetna', 'Cigna', 
                        'UnitedHealthcare', 'Kaiser Permanente'
                    )),
                    'member_id': fake.ssn(),
                    'group_number': str(fake.random_number(digits=6, fix_len=True))
                })
            
            df = pd.DataFrame(data)
            df.to_csv(self.patients_file, index=False)
            print(f"✅ Created sample patients file: {self.patients_file}")
            
        except ImportError:
            # Create basic sample data without Faker
            print("⚠️ Faker library not found. Creating basic sample data...")
            basic_data = [
                {
                    'patient_id': 1, 'first_name': 'John', 'last_name': 'Doe',
                    'dob': '1985-03-15', 'is_returning': True, 'email': 'john.doe@email.com',
                    'location': '123 Main St, Anytown, USA', 'phone_number': '555-0123',
                    'insurance_carrier': 'Blue Cross Blue Shield', 'member_id': '123456789', 'group_number': '987654'
                },
                {
                    'patient_id': 2, 'first_name': 'Jane', 'last_name': 'Smith',
                    'dob': '1990-07-22', 'is_returning': False, 'email': 'jane.smith@email.com',
                    'location': '456 Oak Ave, Anytown, USA', 'phone_number': '555-0124',
                    'insurance_carrier': 'Aetna', 'member_id': '987654321', 'group_number': '123456'
                }
            ]
            df = pd.DataFrame(basic_data)
            df.to_csv(self.patients_file, index=False)
            print(f"✅ Created basic patients file: {self.patients_file}")

    def _create_doctor_schedule(self):
        """Create a sample doctor schedule with realistic availability."""
        doctors = ['Dr. Emily Chen', 'Dr. David Rodriguez']
        schedule = []
        current_date = datetime.now() + timedelta(days=1)  # Start from tomorrow
        
        for _ in range(14):  # Two weeks of schedule
            if current_date.weekday() < 5:  # Monday to Friday only
                for doctor in doctors:
                    # Create time slots from 9 AM to 5 PM (30-minute intervals)
                    for hour in range(9, 17):
                        for minute in [0, 30]:
                            # Randomly make some slots unavailable (30% chance)
                            import random
                            random.seed(42)  # For consistent results
                            is_available = random.random() > 0.3
                            
                            schedule.append({
                                'doctor_name': doctor,
                                'date': current_date.strftime('%Y-%m-%d'),
                                'time': f"{hour:02d}:{minute:02d}",
                                'is_available': is_available
                            })
            current_date += timedelta(days=1)
        
        df = pd.DataFrame(schedule)
        df.to_csv(self.schedule_file, index=False)
        print(f"✅ Created doctor schedule: {self.schedule_file}")

    def _create_appointments_file(self):
        """Create an appointments tracking file with proper columns."""
        columns = [
            'appointment_id', 'patient_first_name', 'patient_last_name', 'patient_dob',
            'patient_phone', 'patient_email', 'patient_location', 'doctor_name', 
            'appointment_date', 'appointment_time', 'duration_minutes', 'is_returning_patient',
            'insurance_carrier', 'insurance_member_id', 'insurance_group_number',
            'created_at', 'status', 'cancellation_reason', 'cancelled_at'
        ]
        
        df = pd.DataFrame(columns=columns)
        df.to_csv(self.appointments_file, index=False)
        print(f"✅ Created appointments file: {self.appointments_file}")

    def lookup_patient(self, first_name: str, last_name: str, dob: str) -> bool:
        """
        Looks up a patient to determine if they are a returning patient.
        Returns True if returning patient, False if new patient.
        """
        try:
            if not os.path.exists(self.patients_file):
                return False
                
            df = pd.read_csv(self.patients_file)
            if df.empty:
                return False
                
            # Normalize the date format
            dob_formatted = self._normalize_date_format(dob)
            
            # Search for matching patient
            match = df[
                (df['first_name'].str.lower() == first_name.lower()) &
                (df['last_name'].str.lower() == last_name.lower()) &
                (df['dob'] == dob_formatted)
            ]
            
            if not match.empty:
                return bool(match.iloc[0]['is_returning'])
            
            return False  # New patient if no match found
            
        except Exception as e:
            print(f"❌ Error looking up patient: {e}")
            return False

    def _normalize_date_format(self, dob: str) -> str:
        """Normalizes common date formats to YYYY-MM-DD."""
        # Try different date formats
        for fmt in ('%m/%d/%Y', '%m-%d-%Y', '%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y'):
            try:
                return datetime.strptime(dob.strip(), fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        # If no format matches, return as is
        return dob.strip()

    def get_available_slots(self, doctor: Optional[str] = None, duration: int = 30) -> List[Dict]:
        """
        Gets a list of available appointment slots.
        Returns up to 8 available slots.
        """
        try:
            if not os.path.exists(self.schedule_file):
                return []
                
            df = pd.read_csv(self.schedule_file)
            if df.empty:
                return []
            
            # Create datetime column for filtering
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
            
            # Filter for future slots only
            now = datetime.now()
            future_slots = df[df['datetime'] > now].copy()
            
            # Filter for available slots
            available = future_slots[future_slots['is_available'] == True].copy()
            
            # Filter by doctor if specified
            if doctor:
                available = available[available['doctor_name'] == doctor]
            
            # Sort by date and time
            available = available.sort_values('datetime')
            
            # Return first 8 slots as list of dictionaries
            result = available.head(8).drop('datetime', axis=1).to_dict('records')
            
            return result
            
        except Exception as e:
            print(f"❌ Error getting available slots: {e}")
            return []

    def _add_new_patient(self, patient_info: Dict, insurance_info: Dict):
        """Adds a new patient record to the patients.csv file."""
        try:
            # Read existing patients or create new DataFrame
            try:
                df = pd.read_csv(self.patients_file)
                new_patient_id = df['patient_id'].max() + 1 if not df.empty else 1
            except (FileNotFoundError, pd.errors.EmptyDataError):
                df = pd.DataFrame()
                new_patient_id = 1
            
            # Create new patient record
            new_patient = {
                'patient_id': new_patient_id,
                'first_name': patient_info.get('first_name'),
                'last_name': patient_info.get('last_name'),
                'dob': patient_info.get('dob'),
                'is_returning': True,  # Will be returning after first visit
                'email': patient_info.get('email'),
                'location': patient_info.get('location'),
                'phone_number': patient_info.get('phone_number', 'N/A'),
                'insurance_carrier': insurance_info.get('carrier', 'N/A'),
                'member_id': insurance_info.get('member_id', 'N/A'),
                'group_number': insurance_info.get('group_number', 'N/A')
            }
            
            # Add to DataFrame and save
            df = pd.concat([df, pd.DataFrame([new_patient])], ignore_index=True)
            df.to_csv(self.patients_file, index=False)
            
            print(f"✅ New patient {patient_info.get('first_name')} {patient_info.get('last_name')} added.")
            
        except Exception as e:
            print(f"❌ Error adding new patient: {e}")

    def save_appointment(self, patient_info: Dict, appointment_info: Dict, insurance_info: Dict) -> str:
        """
        Saves appointment details and adds new patients to the system.
        Returns the appointment ID.
        """
        # Add new patient to system if they're not returning
        if not patient_info.get('is_returning'):
            self._add_new_patient(patient_info, insurance_info)
        
        try:
            # Generate unique appointment ID
            appointment_id = str(uuid.uuid4())[:8].upper()
            
            # Create appointment record
            appointment_data = {
                'appointment_id': appointment_id,
                'patient_first_name': patient_info.get('first_name'),
                'patient_last_name': patient_info.get('last_name'),
                'patient_dob': patient_info.get('dob'),
                'patient_phone': patient_info.get('phone_number', 'N/A'),
                'patient_email': patient_info.get('email'),
                'patient_location': patient_info.get('location'),
                'doctor_name': appointment_info.get('doctor_name'),
                'appointment_date': appointment_info.get('date'),
                'appointment_time': appointment_info.get('time'),
                'duration_minutes': appointment_info.get('duration'),
                'is_returning_patient': patient_info.get('is_returning'),
                'insurance_carrier': insurance_info.get('carrier'),
                'insurance_member_id': insurance_info.get('member_id'),
                'insurance_group_number': insurance_info.get('group_number'),
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'Confirmed',
                'cancellation_reason': '',
                'cancelled_at': ''
            }
            
            # Read existing appointments or create new DataFrame
            try:
                df = pd.read_csv(self.appointments_file)
            except (FileNotFoundError, pd.errors.EmptyDataError):
                df = pd.DataFrame()
            
            # Add appointment and save
            df = pd.concat([df, pd.DataFrame([appointment_data])], ignore_index=True)
            df.to_csv(self.appointments_file, index=False)
            
            # Update schedule to mark slot as unavailable
            self._update_schedule_availability(
                appointment_info.get('doctor_name'),
                appointment_info.get('date'),
                appointment_info.get('time'),
                is_available=False
            )
            
            print(f"✅ Appointment {appointment_id} saved successfully.")
            return appointment_id
            
        except Exception as e:
            print(f"❌ Error saving appointment: {e}")
            return "ERROR"

    def _update_schedule_availability(self, doctor: str, date: str, time: str, is_available: bool):
        """Updates the doctor's schedule availability for a specific slot."""
        try:
            if not os.path.exists(self.schedule_file):
                return
                
            df = pd.read_csv(self.schedule_file)
            if df.empty:
                return
            
            # Find and update the matching slot
            mask = (
                (df['doctor_name'] == doctor) & 
                (df['date'] == date) & 
                (df['time'] == time)
            )
            
            df.loc[mask, 'is_available'] = is_available
            df.to_csv(self.schedule_file, index=False)
            
            print(f"✅ Schedule updated: {doctor} on {date} at {time} -> {'Available' if is_available else 'Booked'}")
            
        except Exception as e:
            print(f"❌ Error updating schedule: {e}")

    def send_patient_intake_form(self, patient_email: str, patient_name: str) -> bool:
        """
        Sends the new patient intake form PDF via email.
        Improved version with better error handling and fallback options.
        """
        # Get email credentials from environment
        sender_email = os.getenv("EMAIL_USER")
        password = os.getenv("EMAIL_PASSWORD")
        smtp_host = os.getenv("EMAIL_HOST", "smtp.gmail.com")
        smtp_port = int(os.getenv("EMAIL_PORT", "587"))
        
        # Check if email credentials are available
        if not sender_email or not password:
            print("\n⚠️ Email credentials not found in environment variables.")
            print("📧 Running in SIMULATION mode...")
            
            # Simulate email sending for demo purposes
            try:
                # Check if form exists
                form_path = os.path.join(os.path.dirname(__file__), '..', 'forms', 'new_patient_intake_form.pdf')
                
                print(f"""
📧 **EMAIL SENT (SIMULATED)**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
To: {patient_email}
Subject: New Patient Intake Form - {patient_name}
Attachment: new_patient_intake_form.pdf
Status: ✅ Successfully delivered (simulated)

**To enable real email sending:**
1. Set up environment variables in .env file:
   EMAIL_USER=your_email@gmail.com
   EMAIL_PASSWORD=your_app_password
   EMAIL_HOST=smtp.gmail.com
   EMAIL_PORT=587

2. For Gmail, use an App Password instead of your regular password
3. Ensure the PDF form exists at: {form_path}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""")
                
                return True
                
            except Exception as e:
                print(f"❌ Error in email simulation: {e}")
                return False
        
        # Real email sending
        try:
            # Check if form file exists
            form_path = os.path.join(os.path.dirname(__file__), '..', 'forms', 'new_patient_intake_form.pdf')
            
            if not os.path.exists(form_path):
                print(f"❌ Patient intake form not found at: {form_path}")
                print("📝 Creating a placeholder form for demonstration...")
                
                # Create forms directory if it doesn't exist
                os.makedirs(os.path.dirname(form_path), exist_ok=True)
                
                # Create a simple text file as placeholder
                with open(form_path.replace('.pdf', '.txt'), 'w') as f:
                    f.write(f"""NEW PATIENT INTAKE FORM
HealthCare Plus Medical Center

Patient Name: {patient_name}
Email: {patient_email}

Please complete this form and bring it to your appointment.

[This is a placeholder form for demonstration purposes]
""")
                form_path = form_path.replace('.pdf', '.txt')
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = patient_email
            msg['Subject'] = f'New Patient Intake Form - {patient_name}'
            
            # Email body
            body = f"""Dear {patient_name},

Welcome to HealthCare Plus Medical Center! 🏥

As a new patient, please find your intake form attached. To ensure a smooth visit:

✅ Complete the attached form
✅ Bring it to your appointment or submit it online
✅ Arrive 15 minutes early for check-in
✅ Bring a valid ID and insurance card

If you have any questions, please call us at (555) 123-4567.

We look forward to seeing you!

Best regards,
HealthCare Plus Medical Center Team

---
This email was sent automatically by our AI scheduling assistant.
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach the form file
            with open(form_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                
                filename = os.path.basename(form_path)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {filename}'
                )
                msg.attach(part)
            
            # Send the email
            server = smtplib.SMTP(smtp_host, smtp_port)
            server.starttls()
            server.login(sender_email, password)
            text = msg.as_string()
            server.sendmail(sender_email, patient_email, text)
            server.quit()
            
            print(f"✅ Successfully sent intake form email to {patient_email}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to send email: {e}")
            print("💡 Suggestion: Check your email credentials and network connection")
            return False

    def find_appointment_by_patient(self, patient_info: Dict) -> Optional[Dict]:
        """
        Finds a patient's most recent, active appointment.
        Returns the appointment details or None if not found.
        """
        try:
            if not os.path.exists(self.appointments_file):
                return None
                
            df = pd.read_csv(self.appointments_file)
            if df.empty:
                return None
            
            # Filter for confirmed appointments only
            confirmed = df[df['status'] == 'Confirmed']
            if confirmed.empty:
                return None
            
            # Normalize the DOB format for comparison
            dob_normalized = self._normalize_date_format(patient_info.get('dob', ''))
            
            # Find matching appointments
            match = confirmed[
                (confirmed['patient_first_name'].str.lower() == patient_info.get('first_name', '').lower()) &
                (confirmed['patient_last_name'].str.lower() == patient_info.get('last_name', '').lower()) &
                (confirmed['patient_dob'] == dob_normalized)
            ]
            
            if not match.empty:
                # Return the most recent appointment
                latest = match.sort_values(by='created_at', ascending=False).iloc[0]
                return latest.to_dict()
            
            return None
            
        except Exception as e:
            print(f"❌ Error finding appointment: {e}")
            return None

    def cancel_appointment(self, appointment_id: str, reason: str = "") -> bool:
        """
        Cancels an appointment, frees the time slot, and removes new patients if they cancel.
        Returns True if successful, False otherwise.
        """
        try:
            if not os.path.exists(self.appointments_file):
                print(f"❌ Appointments file not found.")
                return False
                
            df_appts = pd.read_csv(self.appointments_file)
            if df_appts.empty:
                print(f"❌ No appointments found.")
                return False
            
            # Find the appointment to cancel
            mask = df_appts['appointment_id'] == appointment_id
            if not mask.any():
                print(f"❌ Could not find appointment ID {appointment_id} to cancel.")
                return False
            
            appointment = df_appts.loc[mask].iloc[0]
            was_new_patient = not appointment['is_returning_patient']
            
            # Update appointment status to cancelled
            df_appts.loc[mask, 'status'] = 'Cancelled'
            if reason:
                df_appts.loc[mask, 'cancellation_reason'] = reason
            df_appts.loc[mask, 'cancelled_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Save updated appointments
            df_appts.to_csv(self.appointments_file, index=False)
            
            # Free up the time slot in the schedule
            self._update_schedule_availability(
                appointment['doctor_name'],
                appointment['appointment_date'],
                appointment['appointment_time'],
                is_available=True
            )
            
            # Remove new patients from the patient list if they cancel their first appointment
            if was_new_patient:
                try:
                    df_patients = pd.read_csv(self.patients_file)
                    patient_mask = (
                        (df_patients['first_name'].str.lower() == appointment['patient_first_name'].lower()) &
                        (df_patients['last_name'].str.lower() == appointment['patient_last_name'].lower()) &
                        (df_patients['dob'] == appointment['patient_dob'])
                    )
                    
                    # Remove the patient record
                    df_patients = df_patients[~patient_mask]
                    df_patients.to_csv(self.patients_file, index=False)
                    
                    print(f"✅ Removed new patient {appointment['patient_first_name']} {appointment['patient_last_name']} from patient database due to cancellation.")
                    
                except Exception as e:
                    print(f"⚠️ Could not remove new patient from database: {e}")
            
            print(f"✅ Appointment {appointment_id} has been successfully cancelled.")
            return True
            
        except Exception as e:
            print(f"❌ Error cancelling appointment: {e}")
            return False

    def update_patient_email(self, first_name: str, last_name: str, dob: str, email: str) -> bool:
        """Updates a patient's email address in the database."""
        try:
            if not os.path.exists(self.patients_file):
                return False
                
            df = pd.read_csv(self.patients_file)
            if df.empty:
                return False
            
            # Find matching patient
            dob_normalized = self._normalize_date_format(dob)
            mask = (
                (df['first_name'].str.lower() == first_name.lower()) &
                (df['last_name'].str.lower() == last_name.lower()) &
                (df['dob'] == dob_normalized)
            )
            
            if mask.any():
                df.loc[mask, 'email'] = email
                df.to_csv(self.patients_file, index=False)
                print(f"✅ Updated email for {first_name} {last_name}")
                return True
            else:
                print(f"❌ Patient not found for email update: {first_name} {last_name}")
                return False
                
        except Exception as e:
            print(f"❌ Error updating patient email: {e}")
            return False

    def get_appointments_report(self) -> pd.DataFrame:
        """
        Returns a DataFrame with all appointments for reporting purposes.
        Useful for analytics and monitoring.
        """
        try:
            if os.path.exists(self.appointments_file):
                return pd.read_csv(self.appointments_file)
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"❌ Error reading appointments report: {e}")
            return pd.DataFrame()

    def get_patient_count(self) -> Dict[str, int]:
        """Returns count of returning vs new patients."""
        try:
            if not os.path.exists(self.patients_file):
                return {"returning": 0, "new": 0, "total": 0}
                
            df = pd.read_csv(self.patients_file)
            if df.empty:
                return {"returning": 0, "new": 0, "total": 0}
            
            returning = len(df[df['is_returning'] == True])
            new = len(df[df['is_returning'] == False])
            
            return {
                "returning": returning,
                "new": new,
                "total": len(df)
            }
            
        except Exception as e:
            print(f"❌ Error getting patient count: {e}")
            return {"returning": 0, "new": 0, "total": 0}