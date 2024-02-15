import json
import argparse


parser = argparse.ArgumentParser(description='json reader')
parser.add_argument('-p', '--path', type=str, default='./Results/prediction.json')

args = parser.parse_args()
json_path = args.path

def main():
    with open(json_path, "r") as j:
        data = json.load(j)
        cnt = 0
        for post in data:
            if cnt < 5:
                q = post["query_content"]
                a = post["responses"][0]["content_en"]
                print(f"Query: {q} \n"
                      f"Answer: {a} \n")
                cnt += 1
            else:
                break
    return


if __name__ == "__main__":
    main()
