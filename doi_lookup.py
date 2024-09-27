import win32clipboard
import json
import requests
import sys


def get_bib_entry(doi):
    print("https://api.crossref.org/works/" + doi)
    r = requests.get("https://api.crossref.org/works/" + doi)
    data = json.loads(r.content.decode('utf-8'))

    title = str(data["message"]["title"][0])
    authors = ""
    for author in data["message"]["author"]:
        if len(authors) >0:
            authors += ", "
        authors += str(author["given"]) + " " + str(author["family"])
    try:
        volume= str(data["message"]["volume"])
    except:
        volume=""
    
    i = 1
    while True:
        try:
            if i == 1:
                page= str(data["message"]["page"])
                break
            elif i==2:
                page=str(data['message']['reference'][0]['first-page'])
                break
            elif i==3:
                page=str(data['message']['article-number'])
                break
            else:
                page = "not found"
                break
                
        except:
            i += 1
            continue
        
            
    journal=str(data["message"]["container-title"][0])
    link=str(data["message"]["URL"])
    try:
        year = str(data["message"]["published-print"]["date-parts"][0][0])
    except:
        try:
            year = str(data["message"]["created"]["date-parts"][0][0])
        except:
            year = "not found"
           
    bib_string  = "\"" + title +"\"\n"
    #bib_string  = title +"\n"
    bib_string += authors + "\n"
    if volume != "":
        bib_string += journal + " " + volume + ", " + page + " (" + year +")\n"
    else:
        bib_string += journal + " doi:" + doi + " (" + year +")\n" 
    bib_string += link
    return bib_string

def copy_string(s):
    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardText(s)    
    win32clipboard.CloseClipboard()    

def doi_lookup(doi):
    s = get_bib_entry(doi)
    copy_string(s.encode("mbcs", "ignore"))
    print(s)


if __name__ == "__main__":
    doi = sys.argv[1]
    doi_lookup(doi)
    

    #print(get_bib_entry('10.1002/adpr.202000151'))