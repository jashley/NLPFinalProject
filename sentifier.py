import os
import xml.etree.ElementTree as ET
import string

def main():
    for filename in os.listdir('corpus/fulltext'):
        if ('.xml' in filename):
            writeSentences(filename)

def writeSentences(filename, cleaned = True):
    f = open('corpus/fulltext/'+filename)
    tree = ET.parse(f)
    root = tree.getroot()
    sentences = root.find('sentences').findall('sentence')
    los = []
    for s in sentences:
        if (len(s.text) > 0):
            if cleaned:
                text = cleanSentence(s.text)
            else:
                text = s.text
            los.append(text)
    g = open('sentences/'+filename, 'w+')
    for sent in los:
        g.write(sent + "#SENTENCETOKEN#")
    f.close()
    g.close()
    print("parsed "+filename)
    return los


def getCleanedSentences(filename):
    """quickly read already cleaned files"""
    f = open('sentences/'+filename)
    sentences = f.read().split("#SENTENCETOKEN#")
    los = []
    for s in sentences:
        if (len(s) > 0):
            los.append(s)
    f.close()
    return los


def cleanSentence(text):
    """ remove punctuation and set to lowercase"""
    sent = []
    exclude = set(string.punctuation + '\n')
    for word in text.split(" "):
        #may want to do better things
        # with punctuation
        s = ''.join(ch.lower() for ch in word if ch not in exclude)
        sent += [s]
    return " ".join(sent)

def cleanSentenceKeepPunctuation(text):
    """ remove punctuation and set to lowercase"""
    sent = []
    for word in text.split():
        s = ''.join(ch.lower() for ch in word)
        sent += [s]
    return " ".join(sent)
    
            

    

