from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("reports", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="medicalreport",
            name="notes",
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name="medicalreport",
            name="report_date",
            field=models.DateField(blank=True, null=True),
        ),
    ]
