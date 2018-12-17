# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

# Create your models here.
class Stock(models.Model):
    code = models.CharField(max_length=100)
    name = models.CharField(max_length=100)
    industry = models.CharField(max_length=100)
    area = models.CharField(max_length=100)

    class Meta:
        ordering = ('code',)

    def __unicode__(self):
        return self.code