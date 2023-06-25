from NamedEntityRecognition.extract_entites import extract_entites
from RelationExtraction.extract_relation import extract_relation

if __name__ == '__main__':
    text = '长虹电器股份有限公司，简称长虹。长虹电器股份有限公司总部位于绵阳市高新区，创始人是倪润峰，董事长是赵勇。'
    sentences = text.split('。')

    sentences = [sentence for sentence in sentences if sentence != '']
    triplets = []

    for sentence in sentences:
        entities = []
        l = extract_entites(sentence)
        for x in l:
            if x not in entities:
                entities.append(x)
    
        for i in range(0, len(entities)):
            for j in range(i + 1, len(entities)):
                # if entities[i][1] != entities[j][1]:
                triplets.append((entities[i][0], entities[j][0], extract_relation(sentence, entities[i][0], entities[j][0])))
    
    for triplet in triplets:
        print(triplet)

