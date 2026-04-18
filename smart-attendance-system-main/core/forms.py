from django import forms
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from .models import Lecture, Student, Classroom, Teacher, Timetable, Subject, Room, CancelledLecture
import datetime


class StudentRegistrationForm(forms.Form):
    """Registration form for new students"""
    name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Full Name',
            'autofocus': True
        })
    )
    division = forms.ChoiceField(
        choices=[],
        widget=forms.Select(attrs={
            'class': 'form-control',
        })
    )
    roll_no = forms.CharField(
        max_length=20,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Roll Number (e.g. 001)'
        })
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Password'
        })
    )
    confirm_password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Confirm Password'
        })
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        divisions = Classroom.objects.all().values_list('name', 'name')
        self.fields['division'].choices = [('', 'Select Division')] + list(divisions)

    def clean(self):
        cleaned_data = super().clean()
        division = cleaned_data.get('division')
        roll_no = cleaned_data.get('roll_no')
        password = cleaned_data.get('password')
        confirm_password = cleaned_data.get('confirm_password')

        if password and confirm_password and password != confirm_password:
            raise forms.ValidationError('Passwords do not match.')

        if division and roll_no:
            full_roll_no = f"{division}-{roll_no.zfill(3)}"
            if Student.objects.filter(roll_no=full_roll_no).exists():
                raise forms.ValidationError('A student with this Division and Roll Number already exists.')
            username = f"student_{full_roll_no}"
            if User.objects.filter(username=username).exists():
                raise forms.ValidationError('An account with this Division and Roll Number already exists.')

        return cleaned_data


class TeacherRegistrationForm(forms.Form):
    """Registration form for new teachers"""
    name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Full Name',
            'autofocus': True
        })
    )
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Email Address'
        })
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Password'
        })
    )
    confirm_password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Confirm Password'
        })
    )

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if Teacher.objects.filter(email=email).exists():
            raise forms.ValidationError('A teacher with this email already exists.')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError('An account with this email already exists.')
        return email

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get('password')
        confirm_password = cleaned_data.get('confirm_password')
        if password and confirm_password and password != confirm_password:
            raise forms.ValidationError('Passwords do not match.')
        return cleaned_data


class StudentLoginForm(forms.Form):
    """Custom login form for students using Division + Roll No"""
    division = forms.ChoiceField(
        choices=[],
        widget=forms.Select(attrs={
            'class': 'form-control',
            'id': 'division'
        })
    )
    roll_no = forms.CharField(
        max_length=20,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter Roll Number',
            'autofocus': True
        })
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Password'
        })
    )
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Populate division choices from database
        divisions = Classroom.objects.all().values_list('name', 'name')
        self.fields['division'].choices = [('', 'Select Division')] + list(divisions)
    
    def clean(self):
        cleaned_data = super().clean()
        division = cleaned_data.get('division')
        roll_no = cleaned_data.get('roll_no')
        password = cleaned_data.get('password')
        
        if division and roll_no and password:
            try:
                # Roll number stored as division-rollno (e.g., CS-A-001)
                full_roll_no = f"{division}-{roll_no.zfill(3)}"
                student = Student.objects.get(classroom__name=division, roll_no=full_roll_no)
                user = authenticate(username=student.user.username, password=password)
                if user is None:
                    raise forms.ValidationError('Invalid password')
                cleaned_data['user'] = user
                cleaned_data['student'] = student
            except Student.DoesNotExist:
                raise forms.ValidationError('No student found with this Division and Roll Number')
        
        return cleaned_data


class TeacherLoginForm(forms.Form):
    """Login form for teachers using email"""
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your email',
            'autofocus': True
        })
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Password'
        })
    )
    
    def clean(self):
        cleaned_data = super().clean()
        email = cleaned_data.get('email')
        password = cleaned_data.get('password')
        
        if email and password:
            try:
                teacher = Teacher.objects.get(email=email)
                if not teacher.user:
                    raise forms.ValidationError('Teacher account not set up. Contact admin.')
                user = authenticate(username=teacher.user.username, password=password)
                if user is None:
                    raise forms.ValidationError('Invalid password')
                cleaned_data['user'] = user
                cleaned_data['teacher'] = teacher
            except Teacher.DoesNotExist:
                raise forms.ValidationError('No teacher found with this email')
        
        return cleaned_data


class ScheduleExtraLectureForm(forms.Form):
    """Form for scheduling extra lectures"""
    classroom = forms.ModelChoiceField(
        queryset=Classroom.objects.all(),
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    subject = forms.ModelChoiceField(
        queryset=Subject.objects.all(),
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    room = forms.ModelChoiceField(
        queryset=Room.objects.all(),
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    date = forms.DateField(
        widget=forms.DateInput(attrs={'class': 'form-control', 'type': 'date'})
    )
    start_time = forms.TimeField(
        widget=forms.TimeInput(attrs={'class': 'form-control', 'type': 'time'})
    )
    end_time = forms.TimeField(
        widget=forms.TimeInput(attrs={'class': 'form-control', 'type': 'time'})
    )
    
    def clean(self):
        cleaned_data = super().clean()
        start_time = cleaned_data.get('start_time')
        end_time = cleaned_data.get('end_time')
        date = cleaned_data.get('date')
        classroom = cleaned_data.get('classroom')
        room = cleaned_data.get('room')
        
        if start_time and end_time and start_time >= end_time:
            raise forms.ValidationError('End time must be after start time')
        
        if date and date < datetime.date.today():
            raise forms.ValidationError('Cannot schedule lectures in the past')
        
        # Check for conflicts with existing timetable
        if date and start_time and end_time and classroom:
            day_of_week = date.weekday()
            
            # Get IDs of cancelled lectures for this date
            cancelled_ids = CancelledLecture.objects.filter(
                date=date
            ).values_list('timetable_id', flat=True)
            
            # Check recurring lectures (excluding cancelled ones)
            recurring_conflicts = Timetable.objects.filter(
                classroom=classroom,
                day_of_week=day_of_week,
                is_recurring=True,
                start_time__lt=end_time,
                end_time__gt=start_time
            ).exclude(id__in=cancelled_ids)
            
            if recurring_conflicts.exists():
                conflict = recurring_conflicts.first()
                raise forms.ValidationError(
                    f'{classroom} already has {conflict.subject.name} at {conflict.start_time.strftime("%H:%M")}-{conflict.end_time.strftime("%H:%M")}'
                )
            
            # Check extra lectures for this specific date
            extra_conflicts = Timetable.objects.filter(
                classroom=classroom,
                is_recurring=False,
                extra_date=date,
                start_time__lt=end_time,
                end_time__gt=start_time
            )
            
            if extra_conflicts.exists():
                conflict = extra_conflicts.first()
                raise forms.ValidationError(
                    f'{classroom} already has an extra lecture ({conflict.subject.name}) scheduled at this time on {date}'
                )
        
        # Check for room conflicts
        if date and start_time and end_time and room:
            day_of_week = date.weekday()
            
            # Get IDs of cancelled lectures for this date
            cancelled_ids = CancelledLecture.objects.filter(
                date=date
            ).values_list('timetable_id', flat=True)
            
            # Check recurring room conflicts (excluding cancelled)
            room_recurring_conflicts = Timetable.objects.filter(
                room=room,
                day_of_week=day_of_week,
                is_recurring=True,
                start_time__lt=end_time,
                end_time__gt=start_time
            ).exclude(id__in=cancelled_ids)
            
            if room_recurring_conflicts.exists():
                conflict = room_recurring_conflicts.first()
                raise forms.ValidationError(
                    f'{room} is already booked for {conflict.classroom} ({conflict.subject.name}) at this time'
                )
            
            # Check extra lecture room conflicts for this specific date
            room_extra_conflicts = Timetable.objects.filter(
                room=room,
                is_recurring=False,
                extra_date=date,
                start_time__lt=end_time,
                end_time__gt=start_time
            )
            
            if room_extra_conflicts.exists():
                conflict = room_extra_conflicts.first()
                raise forms.ValidationError(
                    f'{room} is already booked for an extra lecture ({conflict.classroom}) at this time on {date}'
                )
        
        return cleaned_data


class PhotoUploadForm(forms.Form):
    """Form for uploading student photos"""
    photo = forms.ImageField(
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': 'image/*',
            'capture': 'user'  # Prefer front camera on mobile
        })
    )
    photo_type = forms.ChoiceField(
        choices=[
            ('straight', 'Front Facing'),
            ('left', 'Looking Left'),
            ('right', 'Looking Right'),
        ],
        widget=forms.HiddenInput()
    )


class StartLectureForm(forms.Form):
    """Form to start a lecture"""
    lecture_id = forms.IntegerField(widget=forms.HiddenInput())
