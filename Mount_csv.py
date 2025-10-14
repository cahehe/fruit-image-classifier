# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/cahehe/fruit-image-classifier/blob/mount-generate-csv/Mount_csv.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% colab={"base_uri": "https://localhost:8080/"} id="qFKruexfRGYV" outputId="0c49063b-a600-49e9-cdbf-5f7759e30a29"
from google.colab import drive
drive.mount('/content/drive')
# %cd /content/drive/MyDrive/Colab Notebooks/Projects/Fruits Image Classifier/fruit-image-classifier

# %% colab={"base_uri": "https://localhost:8080/"} id="uodHdJxa7FzF" outputId="3398027b-c7e0-4fbb-ddad-f0e2a4e71a99"
import os
import pandas as pd

def generate_csv_from_fruits360(base_dir, output_csv="fruits360_data.csv"):
    """
    Create a CSV with file paths, labels, and split info from Fruits360 dataset.
    - base_dir should point to fruits-360_100x100 (or other variant).
    """
    records = []  # will hold [filepath, label, split]
    seen = set()
    # The dataset has 2 main splits
    for split in ["Training", "Test"]:
        split_dir = os.path.join(base_dir, split)  # e.g. ".../fruits-360_100x100/Training"

        # Each label is a folder, like "Apple Golden 1"
        for label in os.listdir(split_dir):
            label_dir = os.path.join(split_dir, label)

            if not os.path.isdir(label_dir):
                continue  # skip files, we only want folders
            #seen.add("label dir:{}\n".format(label_dir))
            # Loop over all images in that label folder
            for fname in os.listdir(label_dir):
                #print('label_dir:{}  fname:{}'.format(label_dir, fname))
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    #print('label_dir:{}  fname:{}'.format(label_dir, fname))
                    filepath = os.path.join(label_dir, fname)
                    records.append([filepath, label, split])

    # Create dataframe (like Excel table)
    df = pd.DataFrame(records, columns=["filepath", "label", "split"])
    #print("fruits:{}".format(seen))
    #print("records:{}".format(records))
    # Save to CSV
    #path - /content/drive/MyDrive/Colab Notebooks/Projects/Fruits Image Classifier/fruit-image-classifier
    df.to_csv(output_csv, index=False)
    print(f"CSV saved to {output_csv}, total rows: {len(df)}")

    return df


# Example usage: change to match your tree
base_dir = "/content/drive/MyDrive/Colab Notebooks/Projects/Fruits Image Classifier/Fruits-360/fruits-360_100x100/fruits-360"
df = generate_csv_from_fruits360(base_dir)

# Show first 5 rows
print(df.head())


# %% colab={"base_uri": "https://localhost:8080/", "height": 35} id="Z86BcbD6_uU1" outputId="bcc4e813-ec28-4ca1-9f1f-c9db18333fcf"
# %pwd

