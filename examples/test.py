import argparse
from tqdm import tqdm
import pathlib

from kogito.inference import CommonsenseInference
from kogito.models.bart.comet import COMETBART


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str)
    parser.add_argument("--output-dir", type=str)

    args = parser.parse_args()

    with open(args.datapath) as f:
        data = f.readlines()

    csi = CommonsenseInference()
    model = COMETBART.from_pretrained()

    for i, line in tqdm(
        enumerate(data), total=len(data), desc="Generating", position=0, leave=True
    ):
        kgraph = csi.infer(line, model)
        kgraph.to_jsonl(
            f"{args.output_dir}/{pathlib.Path(args.datapath).stem}_{i+1}.json"
        )


if __name__ == "__main__":
    main()
