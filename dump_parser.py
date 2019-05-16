import xml.parsers.expat
import pandas as pd
import sys
import csv
import re
from collections import defaultdict

__version__ = '2.0.0'

Debug = False

csv_separator = ","

page_text_dict = defaultdict(list)
page_text_list = []
page_name_list = []

added_pages = set()


def xml_to_csv(filename):
    # BEGIN xmt_to_csv var declarations
    # Shared variables for parser subfunctions:
    # output_csv, _current_tag, _parent
    # page_id,page_title,page_ns,revision_id,timestamp,contributor_id,contributor_name,bytes_var

    global page_text_dict, page_text_list
    output_csv = None
    _parent = None
    _current_tag = ''
    page_title = bytes_var = ''

    def start_tag(tag, attrs):
        nonlocal output_csv, _current_tag, _parent
        nonlocal bytes_var

        _current_tag = tag

        if tag == 'page' or tag == 'revision' or tag == 'contributor':
            _parent = tag

        if tag == 'upload':
            print("!! Warning: '<upload>' element not being handled", file=sys.stderr)

    def data_handler(data):
        nonlocal output_csv, _current_tag, _parent
        nonlocal page_title, bytes_var

        if _current_tag == '':  # Don't process blank "orphan" data between tags!!
            return

        if _parent:
            if _parent == 'page':
                if _current_tag == 'title':
                    page_title = data
            elif _parent == 'revision':
                if _current_tag == 'text':
                    bytes_var = data
                    page_text_dict[page_title].append(data)

    def end_tag(tag):
        nonlocal output_csv, _current_tag, _parent
        nonlocal page_title, bytes_var

        def has_empty_field(l):
            field_empty = False
            i = 0
            while not field_empty and i < len(l):
                field_empty = (l[i] == '')
                i = i + 1
            return field_empty

        # uploading one level of parent if any of these tags close
        if tag == 'page':
            _parent = None
        elif tag == 'revision':
            _parent = 'page'
        elif tag == 'contributor':
            _parent = 'revision'

        # print revision to revision output csv
        if tag == 'revision':

            revision_row = [page_title, bytes_var]

            # Do not print (skip) revisions that has any of the fields not available
            if not has_empty_field(revision_row):
                output_csv.write(csv_separator.join(revision_row) + '\n')
            else:
                print("The following line has incomplete info and therefore it's been removed from the dataset:")
                print(revision_row)

            # Debug lines to standard output
            if Debug:
                print(csv_separator.join(revision_row))

            # Clearing data that has to be recalculated for every row:
            bytes_var = ''

        _current_tag = ''
    # Very important!!! Otherwise blank "orphan" data between tags remain in _current_tag and trigger data_handler!! >:(

    # BEGIN xml_to_csv body

    # Initializing xml parser
    parser = xml.parsers.expat.ParserCreate()
    input_file = open(filename, 'rb')

    parser.StartElementHandler = start_tag
    parser.EndElementHandler = end_tag
    parser.CharacterDataHandler = data_handler
    parser.buffer_text = True
    parser.buffer_size = 1024

    # writing header for output csv file
    output_csv = open(filename[0:-3] + "csv", 'w', encoding='utf8')
    output_csv.write(csv_separator.join(
        ["page_title", "bytes"]))
    output_csv.write("\n")

    # Parsing xml and writting proccesed data to output csv
    print("Processing...")
    parser.ParseFile(input_file)
    print("Done processing")

    input_file.close()
    output_csv.close()

    return True


def parse_xml(xmlfile):
    print('Dump files to process: {}'.format(xmlfile))
    print("Starting to parse file " + xmlfile)
    if xml_to_csv(xmlfile):
        print("Data dump {} parsed succesfully".format(xmlfile))
    else:
        print("Error: Invalid number of arguments. Please specify one or more .xml file to parse", file=sys.stderr)


def get_links():
    count_1 = 0
    count_2 = 0
    for key in page_text_dict:
        print(key, '\n')
        text = ''.join(page_text_dict[key])
        cut_text = text[:text.find('== See also ==')]
        cut_text = cut_text[:text.find('== Notes ==')]
        cut_text = cut_text[:text.find('== References ==')]
        cut_text = cut_text[:text.find('== Further Reading ==')]
        # print(cut_text)
        links = set(re.findall(r'\[\[(.*?)\]\]', cut_text))
        cleaned_links = set()
        for link in links:
            if not link.startswith(
                    ('Category:', 'Wikipedia:', 'Template:', 'Template talk:', 'Help:', 'MOS:', 'File:', 'Image:',
                     'User:', 'User talk:')):
                link = link.split("|", maxsplit=1)[0]
                cleaned_links.add(link)
        # print([key, list(cleaned_links)])
        with open(r'Dump/cleaned_files/links_{0}.csv'.format(count_2), 'a') as f:
            writer = csv.writer(f)
            writer.writerow([key, list(cleaned_links)])
        count_1 += 1
        if count_1 > 10000:
            print('count 1: {0}, count 2: {1}'.format(count_1, count_2))
            count_2 += 1
            count_1 = 0


def main():
    xmlfile = 'Dump/enwiki-latest-pages-articles.xml'
    # open(r'Dump/links.csv', 'w+').close()
    parse_xml(xmlfile)
    get_links()


if __name__ == '__main__':
    main()
