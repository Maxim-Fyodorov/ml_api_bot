import telebot
from telebot.custom_filters import SimpleCustomFilter
from telebot.types import Message, ReplyKeyboardRemove
from requests import get, post, put, delete
from requests.models import Response
from requests.exceptions import RequestException
from json import loads
from typing import Union, Callable
from pprint import pformat
import re
from pathlib import Path
import pandas as pd
from io import BytesIO
from config import TOKEN, API_ADDRESS, STORAGE_DIR
from keyboards import make_keyboard

bot = telebot.TeleBot(TOKEN)
user_states = {}
api_requests = {}
storage_dir = Path(STORAGE_DIR)
storage_dir.mkdir(exist_ok=True)

train_state = 'training'
retrain_state = 'retraining'
predict_state = 'predicting'

model_stage = 'waiting for model choice'
param_stage = 'waiting for parameters choice'
param_value_stage = 'waiting for parameter value input'
feature_stage = 'waiting for features upload'
target_stage = 'waiting for target upload'

# REPLIES
invalid_model_id = 'This is not valid model ID. Please choose one from the list of available models.'
features_upload = 'Upload the .csv file with features (as retrieved by "pandas.DataFrame.to_csv"). File size should ' \
                  'not exceed 20MB.'
target_upload = 'Upload the .csv file with target (as retrieved by "pandas.Series.to_csv"). File size should ' \
                  'not exceed 20MB.'
param_choice = "Choose desired paramteres. If you had chosen all required parameters, press 'I'm done'"


# CUSTOM MESSAGE HANDLERS
class IsStandard(SimpleCustomFilter):
    key = 'is_standard'

    @staticmethod
    def check(message: Message):
        return message.chat.id not in user_states.keys()


class IsTraining(SimpleCustomFilter):
    key = 'is_training'

    @staticmethod
    def check(message: Message):
        return user_states[message.chat.id]['dialogue'] == train_state


class IsRetraining(SimpleCustomFilter):
    key = 'is_retraining'

    @staticmethod
    def check(message: Message):
        return user_states[message.chat.id]['dialogue'] == retrain_state


class IsPredicting(SimpleCustomFilter):
    key = 'is_predicting'

    @staticmethod
    def check(message: Message):
        return user_states[message.chat.id]['dialogue'] == predict_state


bot.add_custom_filter(IsStandard())
bot.add_custom_filter(IsTraining())
bot.add_custom_filter(IsRetraining())
bot.add_custom_filter(IsPredicting())


class WaitForModelChoice(SimpleCustomFilter):
    key = 'wait_for_model_choice'

    @staticmethod
    def check(message: Message):
        return user_states[message.chat.id]['stage'] == model_stage


class WaitForParam(SimpleCustomFilter):
    key = 'wait_for_param'

    @staticmethod
    def check(message: Message):
        return user_states[message.chat.id]['stage'] == param_stage


class WaitForParamValue(SimpleCustomFilter):
    key = 'wait_for_param_value'

    @staticmethod
    def check(message: Message):
        return user_states[message.chat.id]['stage'] == param_value_stage


class WaitForFile(SimpleCustomFilter):
    key = 'wait_for_file'

    @staticmethod
    def check(message: Message):
        return user_states[message.chat.id]['stage'] in [feature_stage, target_stage]


bot.add_custom_filter(WaitForModelChoice())
bot.add_custom_filter(WaitForParam())
bot.add_custom_filter(WaitForParamValue())
bot.add_custom_filter(WaitForFile())


class ChoiceInChoices(SimpleCustomFilter):
    key = 'choice_in_choices'

    @staticmethod
    def check(message: Message):
        return message.text in user_states[message.chat.id]['choices']


bot.add_custom_filter(ChoiceInChoices())


@bot.message_handler(is_standard=True, commands=['start', 'help'])
def handle_start_help(message: Message) -> None:
    bot.send_message(
        message.chat.id,
        '/help - get the list of available commands\n\n'
        '/get_available_classes - get the list of classes available for training\n'
        '/get_available_params - get the list of parameters available to set\n'
        '/get_models_list - get the list of trained models\n'
        '/train - start the model training dialogue\n'
        '/retrain - start the model retraining dialogue\n'
        '/delete - start the model deleting dialogue\n'
        '/predict - start the predicting dialogue\n'
        '/exit - exit from an active dialogue'
    )


@bot.message_handler(is_standard=True, commands=['get_models_list'])
def handle_get_models_list(message: Message) -> None:
    api_response = api_request(get, 'ml_models')
    if isinstance(api_response, Response):
        bot_response = pformat(loads(api_response.content))
    else:
        bot_response = api_response
    bot.send_message(message.chat.id, bot_response)


@bot.message_handler(is_standard=True, commands=['get_available_classes'])
def handle_get_available_classes(message: Message) -> None:
    api_response = api_request(get, 'classes')
    if isinstance(api_response, Response):
        bot_response = list_prettifier(loads(api_response.content))
    else:
        bot_response = api_response
    bot.send_message(message.chat.id, bot_response)


@bot.message_handler(is_standard=True, commands=['get_available_params'])
def handle_get_available_classes(message: Message) -> None:
    api_response = api_request(get, 'parameters')
    if isinstance(api_response, Response):
        bot_response = dict_of_lists_prettifier(loads(api_response.content))
    else:
        bot_response = api_response
    bot.send_message(message.chat.id, bot_response)


@bot.message_handler(is_standard=True, commands=['train'])
def handle_train(message: Message) -> None:
    api_response = api_request(get, 'classes')
    if isinstance(api_response, Response):
        bot_response = 'Choose the desired model class'
        classes_list = loads(api_response.content)
        reply_markup = make_keyboard(classes_list)
        bot.send_message(message.chat.id, bot_response, reply_markup=reply_markup)
        user_states[message.chat.id] = {
            'dialogue': train_state,
            'stage': model_stage,
            'choices': classes_list
        }
        api_requests[message.chat.id] = {'class': None, 'X': None, 'y': None, 'params': {}}
    else:
        bot.send_message(message.chat.id, api_response)


@bot.message_handler(is_standard=True, regexp='/retrain [0-9]+')
def handle_retrain(message: Message) -> None:
    api_response = api_request(get, 'ml_models')
    if isinstance(api_response, Response):
        input_id = re.search('(?<=/retrain )[0-9]+', message.text)[0]
        models_list = loads(api_response.content).keys()
        if input_id in models_list:
            bot_response = features_upload
            user_states[message.chat.id] = {
                'dialogue': retrain_state,
                'stage': feature_stage,
                'current_choice': input_id
            }
            api_requests[message.chat.id] = {'X': None, 'y': None, }
        else:
            bot_response = invalid_model_id
    else:
        bot_response = api_response
    bot.send_message(message.chat.id, bot_response)


@bot.message_handler(is_standard=True, regexp='/delete [0-9]+')
def handle_delete(message: Message) -> None:
    input_id = re.search('(?<=/delete )[0-9]+', message.text)[0]
    api_response = api_request(delete, 'ml_models/' + input_id)
    if isinstance(api_response, Response):
        if api_response.ok:
            bot_response = f'The model {input_id} was deleted'
        else:
            bot_response = loads(api_response.content)['meta']
    else:
        bot_response = api_response
    bot.send_message(message.chat.id, bot_response)


@bot.message_handler(is_standard=True, regexp='/predict [0-9]+')
def handle_predict(message: Message) -> None:
    api_response = api_request(get, 'ml_models')
    if isinstance(api_response, Response):
        input_id = re.search('(?<=/predict )[0-9]+', message.text)[0]
        models_list = loads(api_response.content).keys()
        if input_id in models_list:
            bot_response = features_upload
            user_states[message.chat.id] = {
                'dialogue': predict_state,
                'stage': feature_stage,
                'current_choice': input_id
            }
            api_requests[message.chat.id] = {'X': None}
        else:
            bot_response = invalid_model_id
    else:
        bot_response = api_response
    bot.send_message(message.chat.id, bot_response)


@bot.message_handler(is_standard=False, is_training=True, wait_for_model_choice=True, choice_in_choices=True)
def handle_model_choice(message: Message):
    api_response = api_request(get, 'parameters')
    if isinstance(api_response, Response):
        chosen_class = message.text
        param_list = loads(api_response.content)[chosen_class]
        bot_response = param_choice
        reply_markup = make_keyboard(param_list + ["I'm done"])
        user_states[message.chat.id]['stage'] = param_stage
        user_states[message.chat.id]['choices'] = param_list
        api_requests[message.chat.id]['class'] = message.text
        bot.send_message(message.chat.id, bot_response, reply_markup=reply_markup)
    else:
        bot.send_message(message.chat.id, api_response)


@bot.message_handler(is_standard=False, is_training=True, wait_for_param=True, choice_in_choices=True)
def handle_param_choice(message: Message):
    bot_response = "Type the desired parameter value. Use English notation with dot for floats. " \
                   "Scientific notation is not supported"
    reply_markup = ReplyKeyboardRemove()
    user_states[message.chat.id]['stage'] = param_value_stage
    user_states[message.chat.id]['current_choice'] = message.text
    bot.send_message(message.chat.id, bot_response, reply_markup=reply_markup)


@bot.message_handler(
    is_standard=False,
    is_training=True,
    wait_for_param=True,
    func=lambda message: message.text == "I'm done"
)
def handle_params_choice(message: Message):
    bot_response = features_upload
    reply_markup = ReplyKeyboardRemove()
    user_states[message.chat.id]['stage'] = feature_stage
    bot.send_message(message.chat.id, bot_response, reply_markup=reply_markup)


@bot.message_handler(is_standard=False, is_training=True, wait_for_param_value=True)
def handle_param_value(message: Message):
    bot_response = param_choice
    reply_markup = make_keyboard(user_states[message.chat.id]['choices'] + ["I'm done"])
    user_states[message.chat.id]['stage'] = param_stage
    api_requests[message.chat.id]['params'][user_states[message.chat.id]['current_choice']] = message.text
    bot.send_message(message.chat.id, bot_response, reply_markup=reply_markup)


# TODO Ошибки со стороны Телеграма?
@bot.message_handler(is_standard=False, is_training=True, wait_for_file=True, content_types=['document'])
@bot.message_handler(is_standard=False, is_retraining=True, wait_for_file=True, content_types=['document'])
@bot.message_handler(is_standard=False, is_predicting=True, wait_for_file=True, content_types=['document'])
def handle_file(message: Message):
    if message.document.file_size > 20000000:
        bot.send_message(message.chat.id, 'The file is too large.')
    else:
        if user_states[message.chat.id]['stage'] == feature_stage:
            api_requests_key = 'X'
        if user_states[message.chat.id]['stage'] == target_stage:
            api_requests_key = 'y'
        file_id = message.document.file_id
        file_object = bot.get_file(file_id)
        url = 'https://api.telegram.org/file/bot' + TOKEN + '/' + file_object.file_path
        r = get(url)
        try:
            if user_states[message.chat.id]['stage'] == feature_stage:
                data = pd.read_csv(BytesIO(r.content), index_col=0)
            if user_states[message.chat.id]['stage'] == target_stage:
                data = pd.read_csv(BytesIO(r.content), index_col=0, squeeze=True)
            api_requests[message.chat.id][api_requests_key] = data.to_dict()
            if user_states[message.chat.id]['stage'] == feature_stage:
                if user_states[message.chat.id]['dialogue'] in ['training', 'retraining']:
                    bot.send_message(message.chat.id, target_upload)
                    user_states[message.chat.id]['stage'] = target_stage
                if user_states[message.chat.id]['dialogue'] == 'predicting':
                    api_response = api_request(
                        get,
                        'ml_models/' + user_states[message.chat.id]['current_choice'] + '/prediction',
                        api_requests[message.chat.id]
                    )
                    if isinstance(api_response, Response):
                        content = loads(api_response.content)
                        if api_response.ok:
                            file_path = storage_dir.joinpath(str(message.chat.id) + '_pred.csv')
                            pred = pd.Series(content)
                            with file_path.open('wb') as file:
                                pred.to_csv(file)
                            with file_path.open('rb') as file:
                                bot.send_document(message.chat.id, data=file, caption='Your prediction')
                            file_path.unlink()
                            clear_user_state(message.chat.id)
                        else:
                            bot.send_message(message.chat.id, exception_prettifier(content['meta']))
                    else:
                        bot.send_message(message.chat.id, api_response)
            elif user_states[message.chat.id]['stage'] == target_stage:
                if user_states[message.chat.id]['dialogue'] == 'training':
                    api_method = post
                    api_hand = 'ml_models'
                if user_states[message.chat.id]['dialogue'] == 'retraining':
                    api_method = put
                    api_hand = 'ml_models/' + user_states[message.chat.id]['current_choice']
                api_response = api_request(api_method, api_hand, api_requests[message.chat.id])
                if isinstance(api_response, Response):
                    content = loads(api_response.content)
                    if api_response.ok:
                        bot.send_message(message.chat.id, pformat(content))
                    else:
                        bot.send_message(message.chat.id, exception_prettifier(content['meta']))
                    clear_user_state(message.chat.id)
                else:
                    bot.send_message(message.chat.id, api_response)
        except (pd.errors.ParserError, UnicodeDecodeError):
            bot.send_message(message.chat.id, 'The file is not in .csv format. Please try again.')


@bot.message_handler(is_standard=False, commands=['exit'])
def handle_exit(message: Message):
    bot.send_message(
        message.chat.id,
        "OK. If you want to continue working with the bot, type the corresponding command.",
        reply_markup=ReplyKeyboardRemove()
    )
    clear_user_state(message.chat.id)


@bot.message_handler(content_types=['text', 'document'])
def handle_possimpoble(message: Message):
    bot.send_message(
        message.chat.id,
        "This is invalid command or input. Follow the instructions. If you want to leave an active dialogue, "
        "type /exit."
    )


def api_request(method: Callable, hand: str, params: dict = {}) -> Union[str, Response]:
    try:
        r = method(
            API_ADDRESS + hand,
            json=params
        )
        return r
    except RequestException:
        return 'The API is not available right now. Please try again later.'


def exception_prettifier(exceptions_dict: dict) -> str:
    text = ''
    for k, v in exceptions_dict.items():
        text += f'Check {k}\n'
        if isinstance(v, dict):
            for k2, v2 in v.items():
              text += f'\t\t\tCheck {k2}:\n\t\t\t\t\t\t{v2[0]}\n'
        else:
            text += v + '\n'
    return text


def list_prettifier(lst: list) -> str:
    return '\n'.join(lst)


def dict_of_lists_prettifier(dct: dict) -> str:
    text = ''
    for k, v in dct.items():
        newline = f'For {k}: {", ".join(v)}\n\n'
        text += newline
    return text


def delete_user_files(chat_id: int, var: str) -> None:
    if 'features_path' in user_states[chat_id].keys():
        Path(user_states[chat_id][var + '_path']).unlink()


def clear_user_state(chat_id: int) -> None:
    if chat_id in user_states.keys():
        delete_user_files(chat_id, 'features')
        delete_user_files(chat_id, 'target')
        del user_states[chat_id]
    if chat_id in api_requests.keys():
        del api_requests[chat_id]


bot.infinity_polling()
