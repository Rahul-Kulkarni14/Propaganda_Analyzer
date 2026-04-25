from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator


LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ur": "Urdu",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "zh-cn": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
}


def get_language_name(language_code):
    return LANGUAGE_NAMES.get(language_code, language_code)


def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def translate_to_english(text):
    translated_text = GoogleTranslator(source="auto", target="en").translate(text)
    return translated_text


def prepare_text_for_analysis(text):
    cleaned_text = text.strip()

    if not cleaned_text:
        return {
            "original_text": text,
            "detected_language": "unknown",
            "detected_language_name": "Unknown",
            "translated_text": "",
            "was_translated": False,
            "error": "No text provided."
        }

    detected_language = detect_language(cleaned_text)

    if detected_language == "en":
        return {
            "original_text": cleaned_text,
            "detected_language": detected_language,
            "detected_language_name": get_language_name(detected_language),
            "translated_text": cleaned_text,
            "was_translated": False,
            "error": None
        }

    try:
        translated_text = translate_to_english(cleaned_text)
        return {
            "original_text": cleaned_text,
            "detected_language": detected_language,
            "detected_language_name": get_language_name(detected_language),
            "translated_text": translated_text,
            "was_translated": True,
            "error": None
        }
    except Exception as error:
        return {
            "original_text": cleaned_text,
            "detected_language": detected_language,
            "detected_language_name": get_language_name(detected_language),
            "translated_text": cleaned_text,
            "was_translated": False,
            "error": str(error)
        }
