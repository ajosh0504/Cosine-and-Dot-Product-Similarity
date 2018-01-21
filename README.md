# Cosine and Dot Product Similarity
 Compare the retrieval performance of two different similarity measures, i.e., dot product and cosine similarity.
# Pre-processing the input files:
The inputs should be such that a folder called 'docs' contains all documents, with one file for each document and a folder called 'queries' with a file for each query.
The folders 'docs' and 'queries' contain examples of how the inputs should look. Ofcourse, in order to actually appreciate the effectiveness of these similarity measures, we would need a large number of docs and queries.

# Cosine Similarity
To know more about cosine similarity, refer the following link:
https://en.wikipedia.org/wiki/Cosine_similarity

# Dot product Similarity
To know more about dot product similarity, refer the following link:
https://nlp.stanford.edu/IR-book/html/htmledition/dot-products-1.html

# Code Description
-vocab: Vocabulary vector obtained by reading each document line by line and creating a list of only first occurences of each word in each of the 500 documents. 

-temp_doc:Array representing frequency with which each word in the vocabulary occurs in each document. It is an array of freqeuncy vectors of all the documents.

-temp_query: An array similar to temp_doc, but for queries.

-dot_sum-To calculate dot product of freqeuncy vectors of document and query

-doc_sum-To calculate dot product of frequency vector of each document with itself.(Equivalent to finding norm)

-query_sum: To calculate dot product of frequency vector of each query with itself.

-S_dot: Contains all the dot products

-S_cos: Contains cosine similarity results

-top_ten_dot: Contains indices of top ten documents using dot product similarity

-top_ten_cos: Contains indices of top ten documents using cosine similarity


