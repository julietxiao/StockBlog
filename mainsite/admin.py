# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.contrib import admin

# Register your models here.
from .models import Stock
# from .models import Stock_change

class StockAdmin(admin.ModelAdmin):
    list_display = ('name', 'code', 'industry', 'area', 'price_change', 'p_change','pe','gpr', 'npr')


admin.site.register(Stock, StockAdmin)
# admin.site.register(Stock_change)