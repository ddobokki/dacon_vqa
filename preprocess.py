import pandas as pd

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

train_df["img_path"] = train_df["image_id"].apply(lambda x: f"data/image/train/{x}.jpg")
test_df["img_path"] = test_df["image_id"].apply(lambda x: f"data/image/test/{x}.jpg")

train_df.to_csv("data/preprocess_train.csv", index=False)
test_df.to_csv("data/preprocess_test.csv", index=False)
