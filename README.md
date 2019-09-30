# Masterarbeit

This Project was created as part of a Master's Thesis at the National Institute of Informatics Tokyo and the Technical University of Munich.

# How to use

1. Download a Wikipedia dump file from https://dumps.wikimedia.org/enwiki/.
2. Run the wiki_dump_parser_own.py to obtain .csv files of the form {article: [links]}.
3. Run the organize_articles.py to split the .csv into smaller, alphabetically ordered files (that can then be opened in excel etc.).
4. In the graph_generator.py, choose an article or category as starting point for the graph, then run it.
5. Run the reduced_google_matrix.py on the generated graph to analyze it.
6. Run the visualize_resulty.py to obtain graphical representations of the results.
