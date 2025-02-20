import spacy
import thulac

zh_tokenizer = thulac.thulac(seg_only=True)

def tokenize_zh(text):
    return zh_tokenizer.cut(text, True).split()

en_tokenizer  = spacy.load('en_core_web_sm')

def tokenize_en(text):
    return [token.text.lower() for token in en_tokenizer(text)]


if __name__ == '__main__':
    print(tokenize_zh('你好世界.'))
    print(tokenize_en('Hello world.'))