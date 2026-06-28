from django.contrib import admin

from .models import KnowledgeDocument, Prediction, Recommendation


admin.site.register(Prediction)
admin.site.register(Recommendation)
admin.site.register(KnowledgeDocument)
