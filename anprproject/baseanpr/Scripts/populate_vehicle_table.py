import random
import string
from django.db import models
from django.core.management.base import BaseCommand
from baseanpr.models import Vehicle  # Import your Vehicle model

# Function to generate random number plates
def generate_random_plate(length=7):
    letters = string.ascii_uppercase
    numbers = string.digits
    plate = ''.join(random.choice(letters + numbers) for _ in range(length))
    return plate

# Populate the Vehicle table with random number plates
def populate_vehicle_table(num_plates=10):
    for _ in range(num_plates):
        plate = generate_random_plate()
        vehicle = Vehicle(plate=plate)
        vehicle.save()

# Call the function to populate the table
populate_vehicle_table()
