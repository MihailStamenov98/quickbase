
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_textual_similarity(schema1, schema2):
    scores = []
    for table1, content1 in schema1.items():
        for table2, content2 in schema2.items():
            table_score = util.pytorch_cos_sim(
                model.encode(table1, convert_to_tensor=True),
                model.encode(table2, convert_to_tensor=True)
            ).item()
            field_scores = []
            for field1 in content1['fields']:
                for field2 in content2['fields']:
                    field_score = util.pytorch_cos_sim(
                        model.encode(field1['name'], convert_to_tensor=True),
                        model.encode(field2['name'], convert_to_tensor=True)
                    ).item()
                    field_scores.append(field_score)
            average_field_score = sum(field_scores) / len(field_scores) if field_scores else 0
            scores.append((table_score + average_field_score) / 2)
    return sum(scores) / len(scores) if scores else 0
