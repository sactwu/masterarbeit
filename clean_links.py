import os
from collections import defaultdict
import csv


def main():
    for filename in os.listdir('Dump/links'):
        number = filename[filename.find('_') + 1:filename.find('.')]
        links = defaultdict()
        cleaned_links = []
        try:
            print(filename)
            with open('Dump/links/{0}'.format(filename), encoding='UTF-8') as f:
                reader = csv.reader(f, skipinitialspace=True, quotechar="\"")
                for row in reader:
                    if not row[1] == []:
                        for link in row[1]:
                            print(link)
                            cleaned_link = link.split("|", maxsplit=1)[0]
                            print(cleaned_links)
                            cleaned_links.append(cleaned_link)
                        # print(row[0], row[1])
                        links[row[0]] = cleaned_links
        except:
            print('error with file {0}'.format(filename))
            continue
        with open('Dump/cleaned_files/links_{0}_cleaned.csv'.format(number), 'wb') as f:
            w = csv.writer(f)
            w.writerows(links.items())
        with open('Dump/cleaned_files/cleaned_links.csv', 'a', encoding='UTF-8') as f:
            w = csv.writer(f)
            w.writerows(links.items())


if __name__ == '__main__':
    main()
