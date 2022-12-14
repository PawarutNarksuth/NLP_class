import re

# Define a a dictionary 'keywords'.
keywords = {'greet': ['hello', 'hi', 'hey'], 
            'goodbye': ['bye','farewell'], 
            'thankyou': ['thank', 'thx']
            }

# Define a dictionary of patterns
patterns = {}

# Iterate over the keywords dictionary
for intent , keys in keywords.items():

    # Create regular expressions and compile them into pattern objects
    patterns[intent] = re.compile('|'.join(keys))
    

# Print the patterns
print(patterns)