import re

# Define a a dictionary 'keywords'.
keywords = {'greet': ['hello', 'hi', 'hey'], 
            'goodbye': ['bye','farewell'], 
            'thankyou': ['thank', 'thx']}

# Define a dictionary of patterns
patterns = {}
# Iterate over the keywords dictionary
for intent , keys in keywords.items():
    
    # Create regular expressions and compile them into pattern objects
    patterns[intent] = re.compile('|'.join(keys))

responses = {'greet': 'Hello you!:)', 
            'goodbye': 'goodbye for now',
            'thankyou': 'you are verywelcome', 
            'default': 'default message'}

# Create templates
bot_template = "BOT : {0}"
user_template = "USER : {0}"

# Define find_name()
def find_name(message):
    name = None
    
    # Create a pattern for checking if the keywords occur
    name_keyword = re.compile('name|call')
    
    name_words = ""
    # Create a pattern for finding capitalized words
    name_pattern = re.compile('[A-Z]{1}[a-z]*')
    
    if name_keyword.search(message):
        # Get the matching words in the string
        name_words = name_pattern.findall(message)
    
    if len(name_words) > 0:
        # Return the name if the keywords are present
        name = ' '.join(name_words)

    return name

# Define respond()
def respond(message):
    # Find the name
    name = find_name(message)
    
    if name is None:
        return "Hi there!"
    else:
        return "Hello, {0}!".format(name)


# Define a function that sends a message to the bot: send_message
def send_message(message):
    
    # Print user_template including the user_message
    print(user_template.format(message))
    
    # Get the bot's response to the message
    response = respond(message)
    
    # Print the bot template including the bot's response.
    print(bot_template.format(response))

# Send messages
send_message("my name is David Copperfield")
send_message("call me Ishmael")
send_message("people call me Cassandra")
send_message("I walk to school")