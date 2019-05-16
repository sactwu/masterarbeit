import networkx as nx
import re
import time
import pickle
import pywikibot
import wikipediaapi
from mediawiki import MediaWiki
from collections import defaultdict


# category = 'Christian Democratic Union of Germany MEPs'
category = 'Karl Carstens'
formatted_category = re.sub(' ', '_', category).lower()
graph = nx.Graph()
human_articles = set()
other_articles = set()
wiki_wiki = wikipediaapi.Wikipedia('en')
all_links = defaultdict(set)
human_links = defaultdict(set)
other_links = defaultdict(set)
articles_in_graph = set()
articles_next = set()
art_id = defaultdict()
count = 0
new_count = 0

wikipedia = MediaWiki(user_agent='pyMediaWiki-ColaresNII')


def is_human(article):
    global human_articles, other_articles, count
    human = False
    if article in human_articles:
        return True
    elif article in other_articles:
        return False
    else:
        try:
            site = pywikibot.Site("en", "wikipedia")
            page = pywikibot.Page(site, article)
            item = pywikibot.ItemPage.fromPage(page)
            item_dict = item.get()
            clm_dict = item_dict["claims"]
            clm_list = clm_dict["P31"]
            for clm in clm_list:
                clm_trgt = clm.getTarget()
                if clm_trgt.getID(numeric=True) == 5:
                    human_articles.add(article)
                    human = True
        except:
            other_articles.add(article)
    other_articles.add(article)
    count += 1
    if count > 200:
        print('saving files...')
        with open('human_articles', 'wb') as fp:
            pickle.dump(human_articles, fp)
        with open('other_articles', 'wb') as fp:
            pickle.dump(other_articles, fp)
        count = 0
    return human


def get_links(article):
    start_time = time.time()
    page_py = wiki_wiki.page(article)
    if page_py.exists():
        for link in page_py.links:
            if not link.startswith(('Wikipedia:', 'Template:', 'Template talk:', 'Help:', 'Category:', 'MOS:', 'File:')):
                all_links[article].add(link)
                if is_human(link):

                    human_links[article].add(link)
                else:
                    other_links[article].add(link)
    print('human articles: {0}'.format(human_links[article]))
    print('other articles: {0}'.format(other_links[article]))
    print(time.time() - start_time)


def add_article_to_graph(article):
    global graph, art_id, articles_in_graph
    if article not in articles_in_graph:
        art_id[article] = len(articles_in_graph)
        graph.add_node(art_id[article])
        articles_in_graph.add(article)
        # add self edge
        graph.add_edge(art_id[article], art_id[article])
        print('graph size is now {0}'.format(len(graph)))


def add_link(article, link):
    global articles_in_graph
    if link not in articles_in_graph:
        add_article_to_graph(link)
    graph.add_edge(art_id[article], art_id[link])


def main():
    global category, articles_in_graph, articles_next, graph, human_articles, other_articles
    print('starting...')
    # read_files()
    try:
        with open('human_articles', 'rb') as fp:
            human_articles = pickle.load(fp)
    except:
        print('there was no human_articles file to be loaded. creating new one...')
    try:
        with open('other_articles', 'rb') as fp:
            other_articles = pickle.load(fp)
    except:
        print('there was no other_articles file to be loaded. creating new one...')
    # print('there was no human_articles file to be loaded. creating new one...')
    # articles_new: articles to be added in the current step

    # articles_new = set(wikipedia.categorymembers(category, results=None, subcategories=True)[0])
    articles_new = {'Karl Carstens'}
    print('elements: ', articles_new)
    current_hop = 0
    num_hops = 10

    while current_hop < num_hops:
        print('starting with hop {0}...'.format(current_hop))
        # add all human articles in the hop to the graph
        for article in articles_new:
            try:
                print('article: {0}'.format(article))
                add_article_to_graph(article)
                # get all links
                get_links(article)
                # is_human for each link
                for link in all_links[article]:
                    is_human(link)
            except:
                continue

        # if human, add to graph and to articles to be added in the next step
        for article in list(articles_in_graph):
            for link in human_links[article]:
                if is_human(link):
                    print('article: {0}'.format(link))
                    add_link(article, link)
                    articles_next.add(link)

        print('articles_next:', articles_next)
        articles_new = articles_next
        articles_next = set()
        print('number of articles to be added in the next step:', len(articles_new))

        # save the graph with the human articles

        temp_graph = graph
        current_articles = articles_in_graph
        nx.write_edgelist(graph, 'Datasets/{0}_{1}_edges_humans.csv'.format(formatted_category, current_hop),
                          delimiter=';')

        # at the end, add all links that are not human
        for article in list(articles_in_graph):
                for link in other_links[article]:
                    add_link(article, link)
        # save the graph with the other articles
        nx.write_edgelist(graph, 'Datasets/{0}_{1}_edges_others.csv'.format(formatted_category, current_hop),
                          delimiter=';')
        articles_in_graph = current_articles
        graph = temp_graph
        print('finished with hop {0}! there are {1} nodes and {2} links in total'
              .format(current_hop, len(graph), graph.number_of_edges()))

        current_hop += 1

    nx.write_edgelist(graph, 'Datasets/{0}_{1}_human_edges.csv'.format(formatted_category, current_hop),
                      delimiter=';')


if __name__ == '__main__':
    main()
