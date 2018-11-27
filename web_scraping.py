from my_data_science import *
from bs4 import BeautifulSoup
from time import sleep
from urllib.request import urlopen
from html.parser import HTMLParser
import requests
import re
import matplotlib.pyplot as plt

def isfloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def get_source(url):
    'returns the source at the URL as a string'
    response = urlopen(url)
    html = response.read()
    return html.decode("utf-8", "backslashreplace")
    
def get_prices(source_string):
    prices = []
    try:
        unclean_prices = re.findall(r'(?<=["][:])\d\d[.]\d\d|(?<=["][:])\d[.]\d\d', source_string)
        prices = [float(i) for i in unclean_prices]
    except ValueError:
        pass
    return prices

def is_video(td):
    pricelabels = td('span', 'pricelabel') 
    return (len(pricelabels) == 1 and pricelabels[0].text.strip().startswith("Video"))
    
def book_info(td):
    """given a BeautifulSoup <td> Tag representing a book, extract the book's details and return a dict"""
    title = td.find("div", "thumbheader").a.text
    by_author = td.find('div', 'AuthorName').text
    authors = [x.strip() for x in re.sub("^By ", "", by_author).split(",")] 
    isbn_link = td.find("div", "thumbheader").a.get("href")
    isbn = re.match("/product/(.*)\.do", isbn_link).groups()[0]
    date = td.find("span", "directorydate").text.strip()
    
    return {
        "title" : title,
        "authors" : authors,
        "isbn" : isbn,
        "date" : date
    }
    
barnes_and_noble = get_source('https://www.barnesandnoble.com/s/data+science?&Ns=P_Sales_Rank%7C0&page=1')
bn_prices = get_prices(barnes_and_noble)
print(bn_prices)
plt.plot(bn_prices)
plt.ylabel('Price of Book')
plt.xlabel('Descending Rank of Data Science Book')
plt.show()
