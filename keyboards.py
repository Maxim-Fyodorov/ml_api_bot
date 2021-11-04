from telebot.types import ReplyKeyboardMarkup, KeyboardButton

def make_keyboard(lst: list) -> ReplyKeyboardMarkup:
    keyboard = ReplyKeyboardMarkup()
    for item in lst:
        keyboard.add(KeyboardButton(item))
    return keyboard