#movies = pd.read_csv('.\\ch6\\train.csv')

# # Find the number of positive and negative reviews
# print('Number of positive and negative reviews: ', movies.label.value_counts())

# # Find the proportion of positive and negative reviews
# print('Proportion of positive and negative reviews: ', movies.label.value_counts() / len(movies))

# length_reviews = movies.text.str.len()

# # How long is the longest review
# print(min(length_reviews))

# Import the required packages
import pandas as pd
from textblob import TextBlob

# text = "how are you"
# # Create a textblob object
# blob_two_cities = TextBlob(text)

# # Print out the sentiment
# print(blob_two_cities.sentiment)

#Read TXT file
f = open(".\\ch6\\titanic.txt", "r")
titanic = f.read()
# Create a textblob object
blob_titanic = TextBlob(titanic)

# Print out its sentiment
print(blob_titanic.sentiment)