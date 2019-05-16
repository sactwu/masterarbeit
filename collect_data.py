from mediawiki import MediaWiki
from wikidata.client import Client
import pickle
import re

wikipedia = MediaWiki(user_agent='pyMediaWiki-ColaresNII')
client = Client()
category = 'Christian Democratic Union of Germany MEPs'
formatted_category = re.sub(' ', '_', category).lower()
articles = set()
categories = set()
new_categories = set()
added_categories = set()
article_list = []


def save_to_files():
    global article_list, articles, categories, new_categories, added_categories
    with open('node_list_{0}'.format(formatted_category), 'wb') as fp:
        pickle.dump(node_list, fp)
    with open('articles_{0}'.format(formatted_category), 'wb') as fp:
        pickle.dump(articles, fp)
    with open('categories_{0}'.format(formatted_category), 'wb') as fp:
        pickle.dump(categories, fp)
    with open('new_categories_{0}'.format(formatted_category), 'wb') as fp:
        pickle.dump(new_categories, fp)
    with open('added_categories_{0}'.format(formatted_category), 'wb') as fp:
        pickle.dump(added_categories, fp)


def read_files():
    global article_list, articles, categories, new_categories, added_categories
    with open('node_list_{0}'.format(formatted_category), 'rb') as fp:
        node_list = pickle.load(fp)
    with open('articles_{0}'.format(formatted_category), 'rb') as fp:
        articles = pickle.load(fp)
    with open('categories_{0}'.format(formatted_category), 'rb') as fp:
        categories = pickle.load(fp)
    with open('new_categories_{0}'.format(formatted_category), 'rb') as fp:
        new_categories = pickle.load(fp)
    with open('added_categories_{0}'.format(formatted_category), 'rb') as fp:
        added_categories = pickle.load(fp)


def get_articles(category):
    global articles, categories, new_categories, added_categories
    arts, subcats = wikipedia.categorymembers(category, results=None, subcategories=True)
    # results=None gets all results
    categories = categories.union(subcats)
    for category in subcats:
        if category not in added_categories:
            new_categories.add(category)
    arts = set(arts)
    # print(arts - articles)
    articles = articles.union(arts)


def main():
    global article_list, category, articles, categories, new_categories, added_categories
    print('starting...')
    # read_files()
    elements = wikipedia.categorymembers(category, results=None, subcategories=True)[0]
    print('elements: ', elements)
    categories.add(category)
    new_categories.add(category)
    count = 0
    while new_categories:

        current_cat = new_categories.pop()
        # print(current_cat)
        get_articles(current_cat)
        added_categories.add(current_cat)
        save_to_files()
        count += 1
        if count % 500 == 0:
            print('count = {0}. There are {1} articles and {2} categories now'.format(count, len(articles), len(categories)))
            save_to_files()
    save_to_files()
    print('finished! there are {0} articles and {1} categories in total'.format(len(articles), len(categories)))


if __name__ == '__main__':
    main()

