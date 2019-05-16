from mediawiki import MediaWiki
import wikipediaapi
import pywikibot
from pywikibot import pagegenerators as pg
import time
site = pywikibot.Site("en", "wikipedia")


wikipedia = MediaWiki(user_agent='pyMediaWiki-ColaresNII')
wiki_wiki = wikipediaapi.Wikipedia('en')

failed_pages = set()
human_articles = set()
other_articles = set()


def is_human(article):
    global failed_pages, human_articles, other_articles
    if article in human_articles:
        return True
    elif article in other_articles:
        return False
    else:
        page = pywikibot.Page(site, article)
        item = pywikibot.ItemPage.fromPage(page)
        item_dict = item.get()
        clm_dict = item_dict["claims"]
        clm_list = clm_dict["P31"]
        for clm in clm_list:
            clm_trgt = clm.getTarget()
            if clm_trgt.getID(numeric=True) == 5:
                human_articles.add(article)
                # print('added {} to human articles'.format(article))
                return True
        else:
            print('no page could be found for {0}'.format(article))
            failed_pages.add(article)
    other_articles.add(article)
    return False


def main():
    article = 'Burkhard Balz'

    start_time = time.time()
    page_py = wiki_wiki.page(article)
    if page_py.exists():

        links = page_py.links
        print(time.time() - start_time)
        for link in links.keys():
            if is_human(link):
                print(links[link])


if __name__ == '__main__':
    main()