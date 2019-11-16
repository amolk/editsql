import argparse
import json

def create_reranker_dataset(pred_file):
  dataset_file = pred_file + ".reranker_dataset.json"

  with open(dataset_file, 'w') as output_f:
    with open(pred_file, 'r') as pred_f:
      for index, line in enumerate(pred_f):
        obj = json.loads(line)
        out = {
          "utterance": " ".join(obj["input_seq"]),
          "instances": obj["beam_info"]["candidates"]
        }

        json.dump(out, output_f)
        output_f.write("\n")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--pred_file', type=str, required=True)
  args = parser.parse_args()

  create_reranker_dataset(args.pred_file)