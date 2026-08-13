from django.db import migrations

import medimind.fields


def encrypt_existing_report_text(apps, schema_editor):
    MedicalReport = apps.get_model("reports", "MedicalReport")
    fields = ("notes", "extracted_text", "summary")
    for report in MedicalReport.objects.iterator(chunk_size=200):
        if any(getattr(report, field) for field in fields):
            report.save(update_fields=list(fields))


class Migration(migrations.Migration):
    dependencies = [("reports", "0002_medicalreport_notes_and_report_date")]

    operations = [
        migrations.AlterField(
            model_name="medicalreport",
            name="notes",
            field=medimind.fields.EncryptedTextField(blank=True),
        ),
        migrations.AlterField(
            model_name="medicalreport",
            name="extracted_text",
            field=medimind.fields.EncryptedTextField(blank=True),
        ),
        migrations.AlterField(
            model_name="medicalreport",
            name="summary",
            field=medimind.fields.EncryptedTextField(blank=True),
        ),
        migrations.RunPython(encrypt_existing_report_text, migrations.RunPython.noop),
    ]
