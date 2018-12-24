# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.test import TestCase

# Create your tests here.

from django.template.loader import get_template
from django.template import Context,RequestContext
from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import redirect
from datetime import datetime
# from utils.util import LSTM
import numpy as np
import random
from stockfun.LSTM import LSTM
from .models import Stock
from django.core.paginator import Paginator,EmptyPage,PageNotAnInteger

