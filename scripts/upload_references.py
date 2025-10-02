import os
import hashlib
import boto3
import dotenv
from pathlib import Path
from botocore.exceptions import NoCredentialsError

dotenv.load_dotenv()

CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_REGION = os.getenv("R2_REGION")
BUCKET = "lfs"
REFERENCE_FOLDER = "tests/reference"
CMAKE_FILE = "tests/reference-images.cmake"
REPO_NAME = "vision.cpp"
LFS_URL = "https://lfs.interstice.cloud"

s3 = boto3.client(
    "s3",
    endpoint_url=f"https://{CF_ACCOUNT_ID}.r2.cloudflarestorage.com",
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name=R2_REGION,
)


def compute_sha256(file_path: Path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def generate_file_id(repo_name, file_path: Path, sha256_hash):
    return f"{repo_name}/{file_path}/{sha256_hash}"


def upload_to_s3(file_path: Path, file_id: str):
    try:
        try:
            s3.head_object(Bucket=BUCKET, Key=file_id)
            return "up-to-date"
        except s3.exceptions.ClientError:
            pass

        s3.upload_file(str(file_path), BUCKET, file_id)
        return "added"
    except NoCredentialsError:
        print("Credentials not available.")
        return "error"


def main():
    cmake_lines = []

    for root, _, files in os.walk(REFERENCE_FOLDER):
        for file in files:
            file_path = Path(root, file)
            relative_path = file_path.relative_to(".").as_posix()

            sha256_hash = compute_sha256(file_path)
            file_id = generate_file_id(REPO_NAME, relative_path, sha256_hash)
            status = upload_to_s3(file_path, file_id)
            print(f"{file_path}: {status}")

            cmake_line = f'file(DOWNLOAD "{LFS_URL}/{file_id}" "{relative_path}" EXPECTED_HASH SHA256={sha256_hash})'
            cmake_lines.append(cmake_line)

    with open(CMAKE_FILE, "w") as cmake_file:
        cmake_file.write("\n".join(cmake_lines))
    print(f"\nCMake code written to {CMAKE_FILE}")


if __name__ == "__main__":
    main()
