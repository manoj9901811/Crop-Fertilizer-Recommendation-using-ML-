from django.db import models

# Create your models here.
class Products(models.Model):
    name=models.CharField(max_length=50)
    price=models.FloatField()
    product_image=models.ImageField(upload_to="Products")
    offer=models.BooleanField(default=False)
    description=models.TextField(null=True)

class Crop(models.Model):
    Nitrogen=models.CharField(max_length=70)
    Phosphorous=models.CharField(max_length=70)
    Potassium=models.CharField(max_length=70)
    Temperature=models.CharField(max_length=70)
    PH=models.CharField(max_length=70)
    Humidity=models.CharField(max_length=70)
    Rainfall=models.CharField(max_length=70)
    Recommend_Crop=models.CharField(max_length=70)
    
from django.db import models

class Fertilizer(models.Model):
    Temperature = models.FloatField()
    Humidity = models.FloatField()
    Moisture = models.FloatField()
    Soil_Type = models.CharField(max_length=50)
    Crop_Type = models.CharField(max_length=50)
    Nitrogen = models.IntegerField()
    Phosphorous = models.IntegerField()
    Potassium = models.IntegerField()
    Recommended_Fertilizer = models.CharField(max_length=50)  # Prediction result

    def __str__(self):
        return f"{self.Crop_Type} - {self.Recommended_Fertilizer}"
