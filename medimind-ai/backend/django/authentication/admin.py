from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from .models import User


@admin.register(User)
class MediMindUserAdmin(UserAdmin):
    fieldsets = UserAdmin.fieldsets + (
        ("Medical identity", {"fields": ("date_of_birth", "gender", "phone_number")}),
    )
    list_display = ("username", "email", "gender", "phone_number", "is_staff", "is_active")
