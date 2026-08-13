from django.db import migrations

import medimind.fields


def encrypt_existing_phone_numbers(apps, schema_editor):
    User = apps.get_model("authentication", "User")
    for user in User.objects.exclude(phone_number="").iterator(chunk_size=500):
        user.phone_number = user.phone_number
        user.save(update_fields=["phone_number"])


class Migration(migrations.Migration):
    dependencies = [("authentication", "0001_initial")]

    operations = [
        migrations.AlterField(
            model_name="user",
            name="phone_number",
            field=medimind.fields.EncryptedTextField(blank=True),
        ),
        migrations.RunPython(encrypt_existing_phone_numbers, migrations.RunPython.noop),
    ]
