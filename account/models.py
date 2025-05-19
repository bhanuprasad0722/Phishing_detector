from django.db import models
from django.contrib.auth.models import User
# Create your models here.
class Movies(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    moviename = models.TextField(max_length=500, blank=True)
    directorname=models.TextField(max_length=100)
    genre=models.TextField(max_length=100)
    rating=models.IntegerField(default=1)
# the tables below this line are the ones that are actually used above this line are the tables not belong to this project 
class Contact(models.Model):
    name = models.TextField(max_length = 100,blank=True)
    email = models.TextField(max_length = 500)
    message = models.TextField(max_length = 1000)

class UrlDataset(models.Model):
    website = models.TextField()
    status = models.CharField(max_length=10)
    page_rank = models.FloatField()
    uploaded_at = models.DateTimeField(auto_now_add=True)


class Prediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    url_input = models.TextField()
    prediction_result = models.CharField(max_length=20)  # phishing / legitimate
    confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} → {self.prediction_result} ({self.confidence:.2f})"

class ModelAccuracy(models.Model):
    model_name = models.CharField(max_length=100)
    accuracy = models.FloatField()
    trained_on = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.model_name} - {self.accuracy:.2f}% @ {self.trained_on.strftime('%Y-%m-%d %H:%M')}"
    
class UserActivity(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    action = models.CharField(max_length=10)  # login / logout
    timestamp = models.DateTimeField(auto_now_add=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)

    def __str__(self):
        return f"{self.user.username} - {self.action} at {self.timestamp}"

