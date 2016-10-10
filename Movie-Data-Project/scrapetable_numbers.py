# Scrapes the distributor market share data from The Numbers
from bs4 import BeautifulSoup
import requests
import csv
from http.client import IncompleteRead

url = 'http://www.the-numbers.com/market/distributors'

movie_links=[]

try:
    response = requests.get(url)
    page = response.text
    soup = BeautifulSoup(page)
except IncompleteRead as e:
    page = e.partial
    
with open("distributor_table.csv", "wt") as f:
    fieldnames = ("rank","distributor","movies","total box office", "tickets","share")
    output = csv.writer(f, delimiter=",")
    output.writerow(fieldnames)
    for tr in soup.find_all('tr')[1:]:
        print(tr)
        tds = tr.find_all('td')
        tds[1]=str(tds[1]).split('">')[1].split('<')[0]
        tds[2]=str(tds[2]).split('">')[1].split('<')[0]
        tds[3]=str(tds[3]).split('">')[1].split('<')[0]
        tds[3]=tds[3].replace('$','')
        tds[3]=tds[3].replace(',','')
        tds[4]=str(tds[4]).split('">')[1].split('<')[0]
        tds[4]=tds[4].replace(',','')
        tds[5]=str(tds[5]).split('">')[1].split('<')[0]
        tds[5]=tds[5].replace('%','')
        output.writerow([tds[0].text,tds[1],tds[2],tds[3],tds[4],tds[5]])