from django.contrib import admin

from .models import Allergy, FamilyHistory, MedicalProfile


admin.site.register(MedicalProfile)
admin.site.register(Allergy)
admin.site.register(FamilyHistory)
