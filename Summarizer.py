from googletrans import Translator
from summarizerAlgo import getrank

f = open("InputText",encoding="utf-8")
bengText = f.read()
f.close()

engText = Translator().translate(bengText,'en')
print(engText.text)
engSummarized = getrank(engText.text)    	#list containing sentences having most rank
#print("\nSummarized text in english ::")
englishSummarizedText = ""
for element in engSummarized:
	englishSummarizedText += element+" "
#print(englishSummarizedText)

finalText = ""
for sentence in engSummarized:
	bengSent = Translator().translate(sentence,'bn')
	temp = bengSent.text
	finalText = finalText+" "+temp

print(finalText)

f = open("SummarizedText",'w', encoding="utf-8")
f.write(finalText)
f.close()


		
