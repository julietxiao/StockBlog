# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models

# Create your models here.

class Stock(models.Model):
    code = models.CharField(max_length=100)
    name = models.CharField(max_length=100)
    industry = models.CharField(max_length=100)
    area = models.CharField(max_length=100)

    p_change = models.CharField(max_length=100)
    price_change = models.CharField(max_length=100)
    # pe 市盈率
    pe = models.CharField(max_length=100)
    # 毛利率(%)
    gpr = models.CharField(max_length=100)
    # npr,净利润率(%)
    npr = models.CharField(max_length=100)


    class Meta:
        ordering = ('code',)

    def __unicode__(self):
        return self.code
#
# # 股票涨跌幅，价格变动
# class Stock_change(models.Model):
#     code = models.CharField(max_length=100)
#     p_change = models.CharField(max_length=100)
#     price_change = models.CharField(max_length=100)
#
#     class Meta:
#         ordering = ('code',)
#
#     def __unicode__(self):
#         return self.code
