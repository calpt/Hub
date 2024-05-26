from glob import glob
import os

from adapters.utils import WEIGHTS_NAME, CONFIG_NAME, HEAD_CONFIG_NAME, HEAD_WEIGHTS_NAME
from huggingface_hub import HfApi
import tqdm
import yaml

from scripts.utils import REPO_FOLDER


OUTPUT_FOLDER = "hf_hub"
HUB_URL = "https://github.com/Adapter-Hub/Hub/blob/master/"


def check_files_exist(api, repo_id, data, version):
    # check if adapter weights & config available
    if not api.file_exists(repo_id=repo_id, filename=WEIGHTS_NAME, revision=version):
        raise ValueError(f"File {WEIGHTS_NAME} not found in version {version}")
    if not api.file_exists(repo_id=repo_id, filename=CONFIG_NAME, revision=version):
        raise ValueError(f"File {CONFIG_NAME} not found in version {version}")

    if data["prediction_head"]:
        # check if head weights & config available
        if not api.file_exists(repo_id=repo_id, filename=HEAD_WEIGHTS_NAME, revision=version):
            raise ValueError(f"File {HEAD_WEIGHTS_NAME} not found in version {version}")
        if not api.file_exists(repo_id=repo_id, filename=HEAD_CONFIG_NAME, revision=version):
            raise ValueError(f"File {HEAD_CONFIG_NAME} not found in version {version}")

    # check readme
    if not api.file_exists(repo_id=repo_id, filename="README.md", revision=version):
        raise ValueError(f"File README.md not found in version {version}")


def verify_migration(
    files,
    hf_org_name: str = "AdapterHub",
):
    api = HfApi()
    errors = []
    for file in tqdm.tqdm(files, desc="Verifying files"):
        try:
            adapter_name = os.path.basename(file).split(".")[0]
            with open(file, "r") as f:
                data = yaml.load(f, yaml.FullLoader)

            repo_id = hf_org_name + "/" + adapter_name
            if not api.repo_exists(repo_id=repo_id):
                raise ValueError("Repo does not exist.")

            # create a subfolder for each version in the output
            for version_data in data["files"]:
                version = version_data["version"]
                is_default = version == data["default_version"]

                check_files_exist(api, repo_id, data, version)
                if is_default:
                    check_files_exist(api, repo_id, data, "main")

        except Exception as e:
            errors.append(file + "\t" + str(e))
            print(f"Error verifying {file}: {e}")

        with open("verify_errors.txt", "w") as f:
            f.write("\n".join(errors))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "folder", type=str, help="Folder containing the adapter files to migrate"
    )
    parser.add_argument("--org_name", type=str, default="AdapterHub")
    args = parser.parse_args()

    files = glob(os.path.join(REPO_FOLDER, args.folder, "*"))
    verify_migration(files, hf_org_name=args.org_name)
