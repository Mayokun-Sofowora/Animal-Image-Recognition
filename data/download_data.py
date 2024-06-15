import os
import kaggle


def download_data():
    """
    Downloads the Animals-10 dataset from Kaggle and unzips it to the 'data/animals10' directory.

    The function uses the Kaggle API to download the specified dataset and unzips the contents
    into the 'data/animals10' directory. If the directory already exists, it skips the download.

    :return: None
    """
    dataset = 'alessiocorrado99/animals10'
    kaggle.api.dataset_download_files(dataset, path='./data/animals10', unzip=True)


if __name__ == "__main__":
    # CHECK IF THE DATA DIRECTORY EXISTS
    if not os.path.exists('./data/animals10/raw-img'):
        # DOWNLOAD THE DATA
        download_data()
    else:
        # PRINT A MESSAGE INDICATING THAT THE DATA DIRECTORY ALREADY EXISTS
        print("Data directory already exists. Skipping download.")
