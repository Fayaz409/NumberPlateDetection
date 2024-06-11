from django.db import models

# Create your models here.
class Vehicle(models.Model):
    plate = models.CharField(max_length=20)
    brand = models.CharField(max_length=20,null=True)
    color = models.CharField(max_length=20,null=True)
    year = models.IntegerField(null=True)