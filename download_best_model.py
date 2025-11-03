# download_best_model.py
# Helper script: replace YOUR_FILE_ID with the Google Drive file id for your best_model.pt
# Autograder will run this prior to evaluation.

import gdown

def download_from_google_drive(file_id, output_path):
    """
    Download a file from Google Drive using its file ID.
    """
    gdown.download(id=file_id, output=output_path, quiet=False)
    return output_path

if __name__ == "__main__":
    # Replace with your actual Google Drive file id
    file_id = "YOUR_FILE_ID"
    output_path = "best_model.pt"
    download_from_google_drive(file_id, output_path)
    print(f"Downloaded to: {output_path}")
