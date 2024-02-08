import json

json_path = "./Results/prediction.json"


def main():
    with open(json_path, "r") as j:
        data = json.load(j)
        cnt = 0
        for post in data:
            if cnt < 5:
                q = post["query_content"][0]
                a = post["response"]["content_en"]
                print(f"Query: {q} \n"
                      f"Answer: {a} \n")
                cnt += 1
            else:
                break
    return


if __name__ == "__main__":
    main()
