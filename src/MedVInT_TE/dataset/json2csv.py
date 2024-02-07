import json
import csv

# Final CSV columns should be like: img_id,img_name,question,answer


def json2csv(json_path, csv_path):
    c = open(csv_path, 'w')
    writer = csv.writer(c)
    writer.writerow(["img_id", "img_name", "question", "answer"])

    j = open(json_path, 'r')
    data = json.load(j)
    for post in data:
        img_id = post["encounter_id"]                                           # Post number
        image_names = post["image_ids"]                                         # List of images
        question = post["query_title_eng"] + " " + post["query_content_en"]     # Query content in English
        answers = post["responses"]                                             # List of answers

        for n in image_names:
            for a in answers:
                writer.writerow([img_id, n, question, a])

    j.close()

    c.close()
