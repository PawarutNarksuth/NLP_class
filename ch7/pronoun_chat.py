import re
def replace_pronouns(message):
    message = message.lower()
    if 'me' in message:
        return re.sub('me','you', message )
    if 'my' in message:
        return re.sub('my','your', message )
    if 'your' in message:
        return re.sub('your','my', message )
    if 'you' in message:
        return re.sub('you','me', message )
    if 'i' in message:
        return re.sub('i','you', message )
    return message
print(replace_pronouns("my last birthday"))
print(replace_pronouns("go with me to Florida"))
print(replace_pronouns("I had my own castle"))
'''ลองเปลี่่ยนฟังชั่นนี้ให้ใน 1 ประโยคจะ replace pronoun ให้ได้มากกว่า 1 ตำแหน่ง'''