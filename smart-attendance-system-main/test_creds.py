import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'attendance_system.settings')
django.setup()
from django.contrib.auth.models import User
from core.models import Student, Teacher, Classroom

# Reset or create teacher
t_user, _ = User.objects.get_or_create(username='admin_teacher', email='teacher@school.com')
t_user.set_password('teacher123')
t_user.save()
Teacher.objects.get_or_create(user=t_user, name='Admin Teacher', email='teacher@school.com')

# Reset or create student
c, _ = Classroom.objects.get_or_create(name='CS-A')
s_user, _ = User.objects.get_or_create(username='student_csa_101')
s_user.set_password('student123')
s_user.save()
Student.objects.get_or_create(user=s_user, roll_no='CS-A-101', name='John Doe', classroom=c)

print('Credentials Created!')
