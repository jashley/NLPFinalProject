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
        g.write(sent)
    f.close()
    g.close()
    print("parsed "+filename)
    return los


def writeSentencesCleanedText(filename):
    f = open('corpus/fulltext/'+filename)
    tree = ET.parse(f)
    root = tree.getroot()
    sentences = root.find('sentences').findall('sentence')
    los = []
    for s in sentences:
        if (len(s.text) > 0):
            los.append(s.text)
    g = open('sentences/'+filename, 'w+')
    for sent in los:
        g.write(sent)
    f.close()
    g.close()
    print("parsed "+filename)
    return los

def cleanSentence(text):

    sent = []
    exclude = set(string.punctuation)
    for word in text.split():
        #may want to do better things
        # with punctuation
        s = ''.join(ch.lower() for ch in word if ch not in exclude)
        sent += [s]
    return " ".join(sent)
    
    
            

    

