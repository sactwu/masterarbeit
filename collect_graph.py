from mediawiki import MediaWiki
import networkx as nx
from wikidata.client import Client
import pickle
import re
from collections import defaultdict
import time

import pywikibot

category = 'Christian Democratic Union of Germany MEPs'
formatted_category = re.sub(' ', '_', category).lower()
article_list = []
failed_pages = set()
page_links = defaultdict()
num_hops = 5
current_hop = -1
articles = [[] for x in range(num_hops + 1)]
art_id = defaultdict()
human_articles = set()
other_articles = set()
human_links = defaultdict(set)
other_links = defaultdict(set)
graph = [nx.Graph() for a in range(num_hops + 1)]
wikipedia = MediaWiki(user_agent='pyMediaWiki-ColaresNII')
client = Client()
site = pywikibot.Site("en", "wikipedia")


def save_to_files(current_hop):
    with open('node_list_{0}_{1}'.format(formatted_category, current_hop), 'wb') as fp:
        pickle.dump(article_list, fp)
    with open('articles_{0}_{1}'.format(formatted_category, current_hop), 'wb') as fp:
        pickle.dump(articles, fp)
    with open('links_{0}_{1}'.format(formatted_category, current_hop), 'wb') as fp:
        pickle.dump(human_links, fp)
    with open('graph_{0}_{1}'.format(formatted_category, current_hop), 'wb') as fp:
        pickle.dump(graph, fp)
    print('successfully saved the files')


def read_files(current_hop):
    global article_list, articles, human_links, graph
    with open('node_list_{0}_{1}'.format(formatted_category, current_hop), 'rb') as fp:
        node_list = pickle.load(fp)
    with open('articles_{0}_{1}'.format(formatted_category, current_hop), 'rb') as fp:
        articles = pickle.load(fp)
    with open('links_{0}_{1}'.format(formatted_category, current_hop), 'rb') as fp:
        human_links = pickle.load(fp)
    with open('graph_{0}_{1}'.format(formatted_category, current_hop), 'rb') as fp:
        graph = pickle.load(fp)
    print('successfully loaded the files')





def separate_links(article, links):
    for link in links:
        # print('link: ', link)
        human = is_human(link)
        if human:
            human_links[article].add(link)
        else:
            other_links[article].add(link)


def get_links(article):
    try:
        page = wikipedia.page(article)
        page_links[article] = page.links
    except:
        print('there was an error involving article {0}'.format(article))


def add_article_to_graph(article):
    global graph, article_list, art_id
    if article not in node_list:
        art_id[article] = len(node_list)
        graph[current_hop].add_node(art_id[article])
        print('article: {0}'.format(article))
        node_list.append(article)


def add_links_to_graph(article, human):
    global graph, page_links
    if human:
        links = human_links[article]
    else:
        links = page_links[article]
    try:
        for link in links:
            if link in article_list:
                link_id = article_list.index(link)
            else:
                link_id = len(article_list)
                article_list.append(link)
            graph[current_hop].add_edge(art_id, link_id)
    except:
        print('there was an error involving article {0}'.format(article))


def main():
    global article_list, category, articles, current_hop
    start_time = time.time()
    print('starting...')
    # read_files()
    elements = wikipedia.categorymembers(category, results=None, subcategories=True)[0]
    print('elements: ', elements)
    for element in elements:
        is_human(element)
    current_hop = 0
    print('starting the loop with {0} articles'.format(len(human_articles)))

    while current_hop < num_hops:
        print('starting with hop {0}...'.format(current_hop))
        # add all human articles in the hop to the graph
        if current_hop < num_hops + 1:
            for article in articles[current_hop]:
                try:
                    get_links(article)
                    separate_links(article, page_links[article])
                    add_article_to_graph(article)
                    # add all links between human articles to the graph
                    add_links_to_graph(article, True)
                except:
                    print('there was an error involving article {0}'.format(article))
            graph[current_hop + 1] = graph[current_hop]
        else:
            for article in articles[current_hop]:
                get_links(article)
                add_article_to_graph(article)
                # add all links between human and all articles to the graph for the last step
                add_links_to_graph(article, False)

        nx.write_edgelist(graph[current_hop], 'Datasets/{0}_{1}_edges.csv'.format(formatted_category, current_hop),
                          delimiter=';')
        save_to_files(current_hop)
        print('finished with hop {0}! there are {1} nodes and {2} links in total'
              .format(current_hop, len(graph[current_hop]), graph[current_hop].number_of_edges()))
        current_hop += 1
    print('finished! It took {0} seconds to run the code'.format(time.time() - start_time))


if __name__ == '__main__':
    main()


























