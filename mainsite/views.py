# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from django.template.loader import get_template
from django.template import Context,RequestContext
from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import redirect
from datetime import datetime

from .models import Stock
from django.core.paginator import Paginator,EmptyPage,PageNotAnInteger

# Create your views here.
def homepage(request):
    template = get_template('index.html')

    stocks = Stock.objects.all()
    # stock_list = list()
    # for count, stock in enumerate(stocks):
    #     stock_list.append("No.{}:".format(str(count)) + str(stock) + "<hr>")
    #     stock_list.append("<small>" + str(stock.name) + "\t" +
    #                       str(stock.industry.encode('utf-8')) + "\t" +
    #                       str(stock.area.encode('utf-8')) + "\t" +
    #                       "</small><br></br>")

    now = datetime.now()
    html = template.render(locals())

    return HttpResponse(html)


def show_stock(request, stock_code):
    template = get_template('stock.html')
    try:
        stock = Stock.objects.get(code=stock_code)
        if stock != None:
            html = template.render(locals())
            return HttpResponse(html)
    except:
        return redirect('/')

    
def show_definition(request):
    template = get_template('definition.html')
    html = template.render(locals())
    return HttpResponse(html)


def show_useKnown(request):
    temmplate=get_template('useKnown.html')
    html=temmplate.render(locals())
    return HttpResponse(html)


def stock_list(request):
    stocks = Stock.objects.all()
    paginator=Paginator(stocks,10)
    p=request.GET.get('p')
    try:
        s=paginator.page(p)
    except PageNotAnInteger:
        s=paginator.page(1)
    except EmptyPage:
        s=paginator.page(paginator.num_pages)
    #request_context=RequestContext(request)
    #request_context.push(locals())
    return render(request,'list.html',locals())