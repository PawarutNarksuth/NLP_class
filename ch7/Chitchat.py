# Define variables
name = "Bot"
weather = "cloudy"

bot_template = "BOT : {0}"

# Define a dictionary with the predefined responses

responses = {
    "what's your name?": "my name is {0}".format(name),
    "what's today's weather?": "the weather is {0}".format(weather),
    "default": "defauclt message"
}

# Return the matching response if there is one, default otherwise
def respond(message):
    # Check if the message is in the responses
    if message in responses:
        # Return the matching message
        bot_message = responses[message]
    else:
        # Return the "default" message
        bot_message = responses["default"]
    return bot_message

def send_message(message):
    # Get the bot's response to the message
    response = respond(message)

    # Print the bot template including the bot's response.
    print(bot_template.format(response))

print(bot_template.format("Hello!"))

value = input("USER : ")
send_message(value)

