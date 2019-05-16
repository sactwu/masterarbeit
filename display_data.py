import pickle

articles = set()
categories = set()
new_categories = set()
added_categories = set()


def read_files():
    global articles, categories
    with open('articles', 'rb') as fp:
        articles = pickle.load(fp)
    with open('categories', 'rb') as fp:
        categories = pickle.load(fp)


def main():
    read_files()
    print(len(articles))
    print(len(categories))


if __name__ == '__main__':
    main()
