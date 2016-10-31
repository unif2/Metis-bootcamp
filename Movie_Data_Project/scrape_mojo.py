# Scrapes all the movies in Box Office mojo. 
# Writes out "title","theaters","release","domestic", "distributor","release","genre","runtime","rating","budget","worldwide" of every foreign movie

from bs4 import BeautifulSoup
import requests
import csv
import re
from http.client import IncompleteRead
import time

domestic_total_regex = re.compile('^[A-Z][a-z]+')

current_url=("http://www.boxofficemojo.com/movies/alphabetical.htm?letter=NUM&p=.html")
movie_links=[]

try:
    response = requests.get(current_url)
    page = response.text
    soup = BeautifulSoup(page)
except IncompleteRead as e:
    page = e.partial
    
letters = soup.findAll('a', href= re.compile('letter='))
letter_index=[] 
for t in letters:
    letter_index.append("http://www.boxofficemojo.com" + t['href'])

# will need to be chuncked. ex. from range(0,3), then range(3,6), etc.
# loop through all the letters
for i in range(0,27): 
    current_url=letter_index[i]
    try:
        response = requests.get(current_url)
        page = response.text
        soup = BeautifulSoup(page)
    except IncompleteRead as e:
        page = e.partial

    navbar=soup.find('div', 'alpha-nav-holder')
    pages = navbar.findAll('a', href= re.compile('alphabetical'))
    page_list=[] 

    for t in pages:
        page_list.append("http://www.boxofficemojo.com" + t['href'])

    movietable=soup.find('div',{'id':'main'})
    movies = movietable.findAll('a', href= re.compile('id='))
    for t in movies:
        movie_links.append("http://www.boxofficemojo.com" + t['href'])

    if pages != None:                 
        i=0    
        while i<len(page_list): 
            current_url=page_list[i] 
            try:
                response = requests.get(current_url)
                page = response.text
                soup = BeautifulSoup(page)
            except IncompleteRead as e:
                page = e.partial

            movietable=soup.find('div',{'id':'main'})
            movies = movietable.findAll('a', href= re.compile('id='))
            for t in movies:
                movie_links.append("http://www.boxofficemojo.com" + t['href'])
            i+=1

# change the nmae of the output file 
with open("movie_data.csv", "wt") as f:
    fieldnames = ("title","theaters","release","domestic", "distributor","release","genre","runtime","rating","budget","worldwide")   
    output = csv.writer(f, delimiter=",")
    output.writerow(fieldnames)

    for url in movie_links:
        if "elizabeth" in url and "elizabethtown" not in url:
            url='http://www.boxofficemojo.com/movies/?id=elizabeth%A0.htm'
        if "simpleplan" in url:
            url='http://www.boxofficemojo.com/movies/?id=simpleplan%A0.htm'
        print (url)
        if url=='http://www.boxofficemojo.com/movies/?id=cinderella81.htm':
            continue
        time.sleep(0.1) 
        current_url = (url + "&adjust_yr=2016&p=.htm")  # all movies in 2016 dollars
           
        try:
            response = requests.get(current_url)
            page = response.text
            soup = BeautifulSoup(page)
        except IncompleteRead as e:
            page = e.partial
                  
        directors = soup.find(text=re.compile('Widest'))
        director_list = []
        if not directors:
            director_list.append('N/A')
        else:
            next_sibling = directors.findNext('td').contents[0]            
            if next_sibling:
                if str(next_sibling)[0:3] == '<fo':
                    temp=str(next_sibling).split('>')[1].split('<')[0]
                else:
                    temp = next_sibling.replace(u'\xa0', ' ')               
                director_list.append(temp)
            else:
                director_list.append('N/A')
        director1=director_list[0]
                
        inrelease = soup.find(text=re.compile('In Rel'))
        inrelease_list = []
        if not inrelease:
            inrelease_list.append('N/A')
        else:
            next_sibling = inrelease.findNext('td').contents[0]            
            print(next_sibling)
            if next_sibling:
                if str(next_sibling)[0:3] == '<fo':
                    temp=str(next_sibling).split('>')[1].split('<')[0]
                else:
                    temp = next_sibling.replace(u'\xa0', ' ') 
                print(temp)
                inrelease_list.append(temp)
            else:
                inrelease_list.append('N/A')
        director2=inrelease_list[0]

        all_bs=soup.findAll('b')
        b_list=[]
        for t in all_bs:
            if 'Domestic Lifetime' not in str(t.encode_contents()):           
                b_list.append(t.encode_contents())
        if len(b_list)>=10:
            if '$'in str(b_list[2]) or 'n/a' in str(b_list[9]):
                if 'n/a' in str(b_list[9]):
                    title=b_list[1]
                    domestic='N/A'
                    if 'N/A' not in str(b_list[2]):
                        distributor=str(b_list[2]).split('>')[1].split('<')[0]
                    else:
                        distributor=b_list[2]
                    if len(str(b_list[3]).split('>'))>3:
                        release=str(b_list[3]).split('>')[2].split('<')[0]
                    else:
                        release=str(b_list[3]).split('>')[1].split('<')[0]
                    genre=b_list[4]
                    runtime=b_list[5]
                    rating=b_list[6]
                    budget=b_list[7]
                    worldwide=b_list[12]
                else:    
                    title=b_list[1]
                    domestic=b_list[2]
                    if 'n/a' not in str(b_list[3]):
                        distributor=str(b_list[3]).split('>')[1].split('<')[0]
                    else:
                        distributor=str(b_list[3])
                    if len(str(b_list[4]).split('>'))>3:
                        release=str(b_list[4]).split('>')[2].split('<')[0]
                    else:
                        release=str(b_list[4]).split('>')[1].split('<')[0]
                    genre=str(b_list[5])
                    runtime=str(b_list[6])
                    rating=str(b_list[7])
                    budget=str(b_list[8])
                    if len(b_list)==11 or '%' not in str(b_list[11]):
                        worldwide='N/A'
                    else:
                        worldwide=str(b_list[13])
                if "Forei" in str(genre): 
                    output.writerow([title,director1,director2,domestic,distributor,release,genre,runtime,rating,budget,worldwide])

