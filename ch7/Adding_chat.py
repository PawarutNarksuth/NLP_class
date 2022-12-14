# Import the random module
import random

name = "Bot"
weather = "cloudy"
bot_template = "BOT : {0}"
user_template = "USER : {0}"

# Define a dictionary containing a list of responses for each message
responses = {
    "what's your name?": [
        "my name is {0}".format(name),
        "they call me {0}".format(name),
        "I am {0}".format(name)
    ],
    "what's today's weather?": [
        "the weather is {0}".format(weather),
        "it's {0} today".format(weather)
    ],

    "default": ["default message"]
}

def respond(message):
    # Check if the message is in the responses
    if message in responses:
        # Return a random matching response
        bot_message = random.choice(responses[message])
    else:
        # Return a random "default" response
        bot_message = random.choice(responses["default"])
    
    return bot_message

def send_message(message):
    # Get the bot's response to the message
    response = respond(message)

    # Print the bot template including the bot's response.
    print(bot_template.format(response))

print(bot_template.format("Hello!"))

for i in range(3):
    value = input("USER : ")
    send_message(value)