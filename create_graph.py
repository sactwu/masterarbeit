import networkx as nx
from mediawiki import MediaWiki
import pickle
import re
import pywikibot
import sys

category = 'Christian Democratic Union of Germany MEPs'
formatted_category = re.sub(' ', '_', category).lower()
articles = set()
article_list = []
failed_pages = []
articles = []
num_hops = 0
graph = nx.Graph()
wikipedia = MediaWiki(user_agent='pyMediaWiki-ColaresNII')
site = pywikibot.Site("en", "wikipedia")


def read_files():
    global articles, article_list
    with open('articles_{0}'.format(formatted_category), 'rb') as fp:
        articles = pickle.load(fp)
    with open('node_list_{0}'.format(formatted_category), 'rb') as fp:
        node_list = pickle.load(fp)


def add_to_graph(article, hop):
    global graph, articles, failed_pages, site
    print(article)
    human = False
    try:
        page = pywikibot.Page(site, article)
        item = pywikibot.ItemPage.fromPage(page)
        item_dict = item.get()
        clm_dict = item_dict["claims"]
        clm_list = clm_dict["P31"]
        for clm in clm_list:
            clm_trgt = clm.getTarget()
            if clm_trgt.getID(numeric=True) == 5:
                human = True
        if human:
            page = wikipedia.page(article)
            links = page.links
            if article not in article_list:
                art_id = len(article_list)
                graph.add_node(art_id)
                article_list.append(article)
            else:
                art_id = article_list.index(article)
            for link in links:
                if link not in article_list:
                    link_id = len(article_list)
                    article_list.append(link)
                    hops[hop].append(link)
                else:
                    link_id = article_list.index(link)
                graph.add_edge(art_id, link_id)
    except:
        print('no page could be found for {0}'.format(article))
        failed_pages.append(article)


def get_graph():
    global articles, article_list, graph, failed_pages, articles

    count = 0
    hop = 0
    for article in articles:
        add_to_graph(article, hop)

        count += 1
        if count % 20 == 0:
            print(count)
            nx.write_edgelist(graph, 'Datasets/{0}_{1}_edges.csv'.format(formatted_category, 'no_hops'), delimiter=';')
    for i in range(num_hops):
        for hop in hops[i]:
            add_to_graph(hop, i + 1)

            count += 1
            if count % 50 == 0:
                print(count)
                nx.write_edgelist(graph, 'Datasets/{0}_{1}_hops_edges.csv'.format(formatted_category, i + 1), delimiter=';')
    print('no page could be found for the following pages:\n{0}'.format(failed_pages))


def main(num_hops_in):
    global category, articles, article_list, articles, num_hops, graph
    num_hops = num_hops_in
    print('starting...')
    hops = [[] for i in range(num_hops + 1)]
    read_files()
    get_graph()
    nx.write_edgelist(graph, 'Datasets/{0}_{1}_edges.csv'.format(formatted_category, num_hops), delimiter=';')
    with open('Datasets/node_list_{0}_{1}_edges'.format(formatted_category, num_hops + 1), 'wb') as fp:
        pickle.dump(node_list, fp)
    print(node_list)
    print('finished! there are {0} nodes with {1} edges in the graph'.format(len(graph), graph.number_of_edges()))


if __name__ == '__main__':
    main(5)
