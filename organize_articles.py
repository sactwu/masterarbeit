import os
import re
import csv
from collections import defaultdict


def main():
    for filename in os.listdir('Dump/cleaned_files/'):
        articles = defaultdict(defaultdict)
        with open('Dump/cleaned_files/{0}'.format(filename), encoding='UTF-8') as f:
            reader = csv.reader(f, skipinitialspace=True, quotechar="\"")
            while True:
                try:
                    row = next(reader)
                except csv.Error:
                    print(filename)
                except StopIteration:
                    break
                article = row[0]
                links = row[1]
                if not article.startswith(
                        ('Category:', 'Wikipedia:', 'Template:', 'Template talk:', 'Help:', 'MOS:', 'File:', 'Image:',
                         'User:', 'User talk:')):
                    if not links == '[]':
                        start_char = article[0].lower()
                        if start_char not in 'abcdefghijklmnopqrstuvwxyz':
                            start_char = '0'
                        articles[start_char][article] = links
        for start_char in articles:
            with open('Dump/ordered_files/{0}.csv'.format(start_char), 'a') as f:
                w = csv.writer(f)
                w.writerows(articles[start_char].items())


if __name__ == '__main__':
    main()