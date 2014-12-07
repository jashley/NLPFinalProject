import os
import xml.etree.ElementTree as ET

def main():
    for filename in os.listdir('corpus/fulltext'):
        if ('.xml' in filename):
            writeSentences(filename)

def writeSentences(filename):
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
    
    
            

    

